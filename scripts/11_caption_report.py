"""Summarise the generated captions (run after 10_merge_captions.py).

Per model and level (image / object):
  coverage, length, empty outputs, CLIPScore (overall and per source), how often the caption
  mentions a material / condition, agreement with the source labels (material and category),
  unsupported-category mentions (hallucination candidates), brand / label-text mentions,
  JSON-format compliance and speed.

Output: data/reports/captions_week2.md, captions_week2.json, caption_examples.jpg
"""
import json
import random
import statistics as st
import textwrap
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont

from captionlib import CAPTIONS, load_caption_cfg, load_generations
from common import PROCESSED, REPORTS, ROOT, SEED, read_jsonl

records = [r for r in read_jsonl(PROCESSED / "manifest_all.jsonl") if r["captions"]]
runtime = {g["caption_id"]: g["runtime_s"] for g in load_generations()}


def mean(xs, nd=3):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), nd) if xs else None


def fmt(v):
    return "–" if v is None else str(v)


def pct(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(map(bool, xs)) / len(xs), 1) if xs else None


img_caps = defaultdict(list)   # model -> [(rec, caption)]
obj_caps = defaultdict(list)   # model -> [(rec, obj, caption)]
for r in records:
    for c in r["captions"]:
        img_caps[c["model"]].append((r, c))
    for o in r["objects"]:
        for c in o["caption_candidates"]:
            obj_caps[c["model"]].append((r, o, c))
models = sorted(set(img_caps) | set(obj_caps), key=lambda m: (m == "template", m))
sources = sorted({r["source"] for r in records})

summary = {"n_images": len(records), "image_ids_file": str((CAPTIONS / "pilot_ids.txt").relative_to(ROOT)),
           "image_level": {}, "object_level": {}}
for m in models:
    rows = img_caps[m]
    a = [c["auto_checks"] for _, c in rows]
    summary["image_level"][m] = {
        "n": len(rows),
        "mean_words": mean([x["n_words"] for x in a], 1),
        "empty_pct": pct([x["empty"] for x in a]),
        "clip_score": mean([c["clip_score"] for _, c in rows]),
        "clip_score_by_source": {s: mean([c["clip_score"] for r, c in rows if r["source"] == s]) for s in sources},
        "mentions_material_pct": pct([x["mentions_material"] for x in a]),
        "material_recall": mean([x["material_recall"] for x in a]),
        "material_conflict_pct": pct([x["material_conflict"] for x in a]),
        "category_recall": mean([x["category_recall"] for x in a]),
        "unsupported_category_pct": pct([bool(x["unsupported_categories"]) for x in a]),
        "mentions_condition_pct": pct([x["mentions_condition"] for x in a]),
        "brand_or_text_pct": pct([x["brand_or_text"] for x in a]),
        "people_animals_pct": pct([x["mentions_people_animals"] for r, c in rows for x in [c["auto_checks"]]
                                   if r["source"] != "coco"]),
        "sec_per_caption": mean([runtime.get(c["caption_id"]) for _, c in rows], 2),
        "top_unsupported": Counter(u for x in a for u in x["unsupported_categories"]).most_common(5),
    }
    rows = obj_caps[m]
    a = [c["auto_checks"] for *_, c in rows]
    summary["object_level"][m] = {
        "n": len(rows),
        "mean_words": mean([x["n_words"] for x in a], 1),
        "empty_pct": pct([x["empty"] for x in a]),
        "clip_score": mean([c["clip_score"] for *_, c in rows]),
        "clip_score_by_source": {s: mean([c["clip_score"] for r, _, c in rows if r["source"] == s]) for s in sources},
        "material_match_pct": pct([x["material_match"] for x in a]),
        "material_conflict_pct": pct([x["material_conflict"] for x in a]),
        "category_match_pct": pct([x["category_match"] for x in a]),
        "mentions_condition_pct": pct([x["mentions_condition"] for x in a]),
        "json_ok_pct": pct([c.get("proposed", {}).get("parsed_json") for *_, c in rows
                            if (c.get("prompt_id") or "").endswith("_json")]),   # only models given a JSON prompt
        "brand_or_text_pct": pct([x["brand_or_text"] for x in a]),
        "sec_per_caption": mean([runtime.get(c["caption_id"]) for *_, c in rows], 2),
        "proposed_conditions": dict(Counter(k for *_, c in rows for k in c.get("proposed", {}).get("condition", []))
                                    .most_common()),
    }

# head-to-head CLIPScore on the same image / object (VLMs only)
vlms = [m for m in models if m not in ("template", "human")]
cfg = load_caption_cfg()
pair = ([m for m in cfg["compare_pair"] if m in vlms][:2] if sum(m in vlms for m in cfg["compare_pair"]) >= 2 else
        [m for m in cfg["pilot_compare_pair"] if m in vlms] if all(m in vlms for m in cfg["pilot_compare_pair"]) else
        vlms[:2])
h2h = {}
if len(pair) == 2:
    a_m, b_m = pair
    for level, groups in (("image", [r["captions"] for r in records]),
                          ("object", [o["caption_candidates"] for r in records for o in r["objects"]])):
        wins = Counter()
        for caps in groups:
            s = {c["model"]: c["clip_score"] for c in caps if c["clip_score"] is not None}
            if a_m in s and b_m in s:
                wins[a_m if s[a_m] > s[b_m] else b_m if s[b_m] > s[a_m] else "tie"] += 1
        h2h[level] = dict(wins)
summary["clip_head_to_head"] = h2h
json.dump(summary, open(REPORTS / "captions_week2.json", "w"), indent=2)

# ---------------- markdown ----------------
L = [f"# Caption generation report (Week 2)\n",
     f"- **Images captioned:** {len(records)} ({', '.join(f'{s} {sum(r['source'] == s for r in records)}' for s in sources)})",
     f"- **Models:** {', '.join(models)}",
     f"- **Image ids:** `{summary['image_ids_file']}`\n",
     "Auto-check columns come from keyword rules in `scripts/captionlib.py`. They are signals for the "
     "human reviewer, not ground truth (see WEEK2_UPDATES.md, section 6).\n",
     "## Image-level captions\n",
     "| metric | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
img_rows = [("captions", "n"), ("mean words", "mean_words"), ("empty %", "empty_pct"),
            ("CLIPScore", "clip_score"), ("mentions a material %", "mentions_material_pct"),
            ("material recall", "material_recall"), ("material conflict %", "material_conflict_pct"),
            ("category recall", "category_recall"), ("unsupported category %", "unsupported_category_pct"),
            ("mentions a condition %", "mentions_condition_pct"), ("brand / label text %", "brand_or_text_pct"),
            ("people / animals mentioned % (TACO+ZeroWaste)", "people_animals_pct"), ("sec / caption", "sec_per_caption")]
for label, k in img_rows:
    L.append(f"| {label} | " + " | ".join(fmt(summary["image_level"][m][k]) for m in models) + " |")
for s in sources:
    L.append(f"| CLIPScore ({s}) | " + " | ".join(fmt(summary["image_level"][m]["clip_score_by_source"][s])
                                               for m in models) + " |")
L += ["", "Most frequent unsupported categories: " + "; ".join(
    f"**{m}** {summary['image_level'][m]['top_unsupported']}" for m in models), ""]
L += ["## Object-level captions (box crops)\n", "| metric | " + " | ".join(models) + " |",
      "|---|" + "---:|" * len(models)]
obj_rows = [("captions", "n"), ("mean words", "mean_words"), ("empty %", "empty_pct"), ("CLIPScore", "clip_score"),
            ("material match %", "material_match_pct"), ("material conflict %", "material_conflict_pct"),
            ("category match %", "category_match_pct"), ("mentions a condition %", "mentions_condition_pct"),
            ("JSON format followed %", "json_ok_pct"), ("brand / label text %", "brand_or_text_pct"),
            ("sec / caption", "sec_per_caption")]
for label, k in obj_rows:
    L.append(f"| {label} | " + " | ".join(fmt(summary["object_level"][m][k]) for m in models) + " |")
for s in sources:
    L.append(f"| CLIPScore ({s}) | " + " | ".join(fmt(summary["object_level"][m]["clip_score_by_source"][s])
                                               for m in models) + " |")
L += ["", "Conditions proposed by the VLMs (object level): " + "; ".join(
    f"**{m}** {summary['object_level'][m]['proposed_conditions'] or '{}'}" for m in vlms), ""]
if h2h:
    L += [f"## Head-to-head CLIPScore ({pair[0]} vs {pair[1]})\n",
          "| level | " + " | ".join([pair[0], pair[1], "tie"]) + " |", "|---|---:|---:|---:|"]
    for level, w in h2h.items():
        L.append(f"| {level} | {w.get(pair[0], 0)} | {w.get(pair[1], 0)} | {w.get('tie', 0)} |")
    L.append("")

# examples: 2 images per source, all models side by side
rng = random.Random(SEED)
examples = []
for s in sources:
    pool = [r for r in records if r["source"] == s]
    examples += rng.sample(pool, min(2, len(pool)))
L += ["## Examples\n", "Rendered in `data/reports/caption_examples.jpg`.\n"]
for r in examples:
    L.append(f"**{r['image_id']}** ({r['source']}; labels: {r['scene']['category_counts']})\n")
    for c in sorted(r["captions"], key=lambda c: models.index(c["model"])):
        L.append(f"- *{c['model']}* (CLIPScore {c['clip_score']}): {c['text']}")
    big = max(r["objects"], key=lambda o: o["bbox"][2] * o["bbox"][3])
    L.append(f"- largest object `{big['object_id']}` ({big['source_category']}):")
    for c in sorted(big["caption_candidates"], key=lambda c: models.index(c["model"])):
        L.append(f"  - *{c['model']}*: {c['text']}")
    L.append("")
(REPORTS / "captions_week2.md").write_text("\n".join(L))

# ---------------- example sheet ----------------
W, IMG_H, TXT_H = 520, 340, 230
font = ImageFont.load_default(size=14)
sheet = Image.new("RGB", (W * 3, (IMG_H + TXT_H) * 2), "white")
draw = ImageDraw.Draw(sheet)
for i, r in enumerate(examples[:6]):
    x0, y0 = (i % 3) * W, (i // 3) * (IMG_H + TXT_H)
    im = Image.open(ROOT / "data" / r["file_name"]).convert("RGB")
    s = min((W - 10) / im.width, (IMG_H - 10) / im.height)
    d = ImageDraw.Draw(im)
    for o in r["objects"]:
        x, y, w, h = o["bbox"]
        d.rectangle([x, y, x + w, y + h], outline=(255, 60, 60), width=max(2, round(3 / s)))
    im = im.resize((round(im.width * s), round(im.height * s)))
    sheet.paste(im, (x0 + 5, y0 + 5))
    ty = y0 + IMG_H
    draw.text((x0 + 8, ty), f"{r['image_id']}  {dict(r['scene']['category_counts'])}"[:70], fill="black", font=font)
    ty += 20
    for c in sorted(r["captions"], key=lambda c: models.index(c["model"])):
        for j, line in enumerate(textwrap.wrap(f"{c['model']}: {c['text']}", 62)[:3]):
            draw.text((x0 + 8, ty), line, fill=(30, 30, 140) if j == 0 else (60, 60, 60), font=font)
            ty += 17
        ty += 4
sheet.save(REPORTS / "caption_examples.jpg", quality=88)
print((REPORTS / "captions_week2.md").read_text()[:3000])
