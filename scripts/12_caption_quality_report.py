"""Caption quality report for one source (default: ZeroWaste) - run after 10_merge_captions.py.

Week 2 feedback: captions need more detail (how many waste items, of which material) and there must be
parameters that measure how good a caption is. For every image-level caption this report uses the
auto checks from captionlib.quality_checks / quality_score:

  count stated %          caption gives a total count (or per-material counts that add up)
  count correct %         that count equals the number of annotated objects
  count error             mean |stated - annotated| when a count is given
  material count acc.     share of annotated materials whose count is stated correctly
  material recall         share of annotated materials that the caption names
  hallucination %         people/animals, non-waste objects, wrong setting, unsupported category, material conflict
  repetitive %            repetition loops ("a plastic cover, a plastic cover, ...")
  detail score            share of {count, material, location, colour, condition, setting} mentioned
  CLIPScore               image-text alignment (reference-free)
  quality (0-100)         weighted combination (weights in configs/captioning.yaml -> quality_score)

Only captions on the same images are compared head to head (the "common images" table).

  python 12_caption_quality_report.py                    # ZeroWaste
  python 12_caption_quality_report.py --source taco

Output: data/reports/caption_quality_<source>.md / .json
"""
import argparse
import json
import statistics as st
from collections import defaultdict

from captionlib import is_hallucinating, load_caption_cfg
from common import PROCESSED, REPORTS, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="zerowaste", help="taco | zerowaste | coco | all")
args = ap.parse_args()

records = [r for r in read_jsonl(PROCESSED / "manifest_all.jsonl")
           if r["captions"] and (args.source == "all" or r["source"] == args.source)]
if not records:
    raise SystemExit(f"no captioned images for source '{args.source}' - run 08-10 first")
cfg = load_caption_cfg()
weights = cfg.get("quality_score", {}).get("weights", {})


def mean(xs, nd=2):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), nd) if xs else None


def pct(xs):
    xs = list(xs)
    return round(100 * sum(map(bool, xs)) / len(xs), 1) if xs else None


def label(model, prompt_id):
    """VLM captions made with a count-aware prompt are reported as their own row."""
    return f"{model} [{prompt_id}]" if prompt_id and prompt_id.endswith(("_zw", "_zw_short")) else model


by_model = defaultdict(list)          # label -> [(rec, caption)]
for r in records:
    for c in r["captions"]:
        by_model[label(c["model"], c.get("prompt_id"))].append((r, c))


def summarise(rows):
    a = [c["auto_checks"] for _, c in rows]
    stated = [x for x in a if x.get("count_correct") is not None]
    return {
        "n": len(rows),
        "quality": mean([x.get("quality_score") for x in a], 1),
        "count_stated_pct": pct(x.get("count_correct") is not None for x in a),
        "count_correct_pct": pct(x.get("count_correct") for x in a),
        "count_error": mean([x["count_error"] for x in stated]),
        "material_count_acc": mean([x.get("material_count_accuracy") for x in a]),
        "material_recall": mean([x.get("material_recall") for x in a]),
        "hallucination_pct": pct(is_hallucinating({**x, "mentions_people_animals":
                                                   x.get("mentions_people_animals") and r["source"] != "coco"})
                                 for (r, _), x in zip(rows, a)),
        "repetitive_pct": pct(x.get("repetitive") for x in a),
        "detail_score": mean([x.get("detail_score") for x in a]),
        "clip": mean([c["clip_score"] for _, c in rows], 3),
        "words": mean([x["n_words"] for x in a], 1),
    }


summary = {m: summarise(rows) for m, rows in by_model.items()}
order = sorted(summary, key=lambda m: -(summary[m]["quality"] or 0))

# head to head on the images every caption source has
common = set.intersection(*({r["image_id"] for r, _ in rows} for rows in by_model.values()))
head = {m: summarise([(r, c) for r, c in rows if r["image_id"] in common]) for m, rows in by_model.items()} \
    if common else {}

json.dump({"source": args.source, "n_images": len(records), "weights": weights, "all_images": summary,
           "common_images": {"n": len(common), "models": head}},
          open(REPORTS / f"caption_quality_{args.source}.json", "w"), indent=2)

COLS = [("n", "captions"), ("quality", "quality (0-100)"), ("count_stated_pct", "count stated %"),
        ("count_correct_pct", "count correct %"), ("count_error", "count error"),
        ("material_count_acc", "material count acc."), ("material_recall", "material recall"),
        ("hallucination_pct", "hallucination %"), ("repetitive_pct", "repetitive %"),
        ("detail_score", "detail score"), ("clip", "CLIPScore"), ("words", "words")]


def table(stats):
    out = ["| model | " + " | ".join(h for _, h in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for m in sorted(stats, key=lambda m: -(stats[m]["quality"] or 0)):
        out.append(f"| {m} | " + " | ".join("–" if stats[m][k] is None else str(stats[m][k]) for k, _ in COLS) + " |")
    return out


def examples(model, k=2):
    rows = sorted(by_model[model], key=lambda rc: rc[1]["auto_checks"].get("quality_score") or 0)
    picks = rows[-k:][::-1] + (rows[:1] if len(rows) > k else [])
    return [f"- `{r['image_id']}` ({len(r['objects'])} objects) **{c['auto_checks'].get('quality_score')}**: "
            f"{c['text'][:300]}" for r, c in picks]


L = [f"# Caption quality report: {args.source}\n",
     f"- **Captioned images:** {len(records)}",
     f"- **Caption sources:** {', '.join(order)}",
     f"- **Quality score weights:** " + ", ".join(f"{k} {v}" for k, v in weights.items()) + "\n",
     "## What each parameter measures\n",
     "| parameter | meaning | good value |", "|---|---|---|",
     "| count stated % | caption gives how many waste items there are | high |",
     "| count correct % | stated count equals the number of annotated objects | high |",
     "| count error | mean difference between stated and annotated count | 0 |",
     "| material count acc. | share of materials (cardboard, plastic, metal) whose count is right | 1 |",
     "| material recall | share of annotated materials named in the caption | 1 |",
     "| hallucination % | mentions people/animals, non-waste objects (keyboard, book), a wrong setting "
     "(table, floor, street) or a category/material that is not annotated | 0 |",
     "| repetitive % | caption contains a repetition loop | 0 |",
     "| detail score | share of {count, material, location, colour, condition, setting} mentioned | 1 |",
     "| CLIPScore | image-text similarity from CLIP ViT-B/32 (text is cut at 77 tokens) | 0.8+ |",
     "| quality (0-100) | weighted mean of count, material, no-hallucination, detail, CLIP, fluency | high |\n",
     f"## All captioned images\n", *table(summary), ""]
if head:
    L += [f"## Same {len(common)} images for every caption source (fair comparison)\n", *table(head), ""]
L += ["## Examples (best two and worst caption per source)\n"]
for m in order:
    L += [f"**{m}**\n", *examples(m), ""]
L += ["## Caveats\n",
      "- Count, material and hallucination checks compare the caption with the annotation using keyword rules. "
      "They are signals for the reviewer, not ground truth.",
      "- `zw-detailed` and `template` are built from the annotation, so their counts are correct by construction. "
      "Their score shows how complete and readable they are, not whether a model can count. "
      "Their colour words come from the median pixel colour inside each object's mask.",
      "- ZeroWaste only annotates cardboard, soft plastic, rigid plastic and metal. White paper sheets and other "
      "unlabelled items on the belt are not counted, so a VLM that counts them is marked wrong.",
      "- Reference-based metrics (BLEU, CIDEr, METEOR) need human-approved captions; they are planned for Week 4 "
      "once reviewers have approved captions in the app."]
(REPORTS / f"caption_quality_{args.source}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(table(summary)))
if head:
    print(f"\nsame {len(common)} images:")
    print("\n".join(table(head)))
print(f"\n-> data/reports/caption_quality_{args.source}.md")
