"""Generate image-level and object-level caption candidates with one or more VLMs.

Image level : the whole image + the model's image prompt.
Object level: a padded crop around each (non-tiny) annotated box + the object prompt,
              so every object caption is linked to its object_id -> bbox / mask.

Raw outputs go to data/captions/<model>.jsonl (one row per candidate). The script is
resumable: candidates whose caption_id already exists are skipped.

Examples
  python 08_generate_captions.py --models blip-base smolvlm-256m template --subset pilot
  python 08_generate_captions.py --models blip2 qwen2.5-vl-7b --subset all --batch-size 8   # GPU
  python 08_generate_captions.py --models qwen2.5-vl-7b --subset split:test --level image --num-candidates 3
"""
import argparse
import json
import random
import time
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

from captionlib import (CAPTIONS, crop_object, fill_prompt, load_caption_cfg, make_caption_row, read_jsonl_safe,
                        template_image_caption, template_object_caption)
from common import PROCESSED, ROOT, SEED, load_taxonomy, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--subset", default="pilot", help="pilot | all | split:<train|val|test> | ids:<file>")
ap.add_argument("--level", default="both", choices=["image", "object", "both"])
ap.add_argument("--limit", type=int, default=None, help="max images (for quick tests)")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--num-candidates", type=int, default=None, help="overrides generation.num_candidates")
ap.add_argument("--device", default=None)
args = ap.parse_args()

cfg = load_caption_cfg()
tax = load_taxonomy()
gen_cfg = cfg["generation"]
crop_cfg = cfg["object_crop"]
n_cand = args.num_candidates or gen_cfg["num_candidates"]
records = read_jsonl(PROCESSED / "manifest_all.jsonl")


def select(records):
    if args.subset == "all":
        return records
    if args.subset.startswith("split:"):
        return [r for r in records if r["split"] == args.subset[6:]]
    if args.subset.startswith("ids:"):
        ids = set(open(args.subset[4:]).read().split())
        return [r for r in records if r["image_id"] in ids]
    # pilot: fixed random sample per source from the configured split(s)
    p = cfg["pilot"]
    rng = random.Random(p.get("seed", SEED))
    out = []
    for src, n in p["images_per_source"].items():
        pool = sorted((r for r in records if r["source"] == src and r["split"] in p["splits"]),
                      key=lambda r: r["image_id"])
        out += rng.sample(pool, min(n, len(pool)))
    (CAPTIONS / "pilot_ids.txt").write_text("\n".join(r["image_id"] for r in out) + "\n")
    return out


def objects_to_caption(rec):
    objs = [o for o in rec["objects"] if not set(o["flags"]) & set(crop_cfg["skip_flags"])]
    objs.sort(key=lambda o: -o["bbox"][2] * o["bbox"][3])
    return objs[:crop_cfg["max_objects_per_image"]]


def gen_params(level):
    return {"max_new_tokens": gen_cfg[f"max_new_tokens_{level}"], "num_candidates": n_cand,
            "do_sample": n_cand > 1, "temperature": gen_cfg["temperature"] if n_cand > 1 else None}


def make_row(model, mcfg, rec, obj, prompt_id, prompt, raw, k, runtime, crop_box=None):
    return make_caption_row(model, mcfg, rec, obj, prompt_id, prompt, raw, k, runtime,
                            gen_params("object" if obj else "image"), crop_box)


subset = select(records)
if args.limit:
    subset = subset[:args.limit]
print(f"{len(subset)} images selected ({args.subset})")

for model in args.models:
    mcfg = cfg["models"][model]
    out_path = CAPTIONS / f"{model}.jsonl"
    done = {r["caption_id"] for r in read_jsonl_safe(out_path)}
    fout = open(out_path, "a")

    # ---- rule-based baseline: no model to load ----
    if mcfg["backend"] == "template":
        n_new = 0
        for rec in subset:
            jobs = []
            if args.level in ("image", "both"):
                jobs.append((None, template_image_caption(rec)))
            if args.level in ("object", "both"):
                jobs += [(o, template_object_caption(o, rec["source"])) for o in objects_to_caption(rec)]
            for obj, text in jobs:
                row = make_row(model, mcfg, rec, obj, "template", None, text, 0, 0.0)
                if row["caption_id"] not in done:
                    fout.write(json.dumps(row) + "\n")
                    n_new += 1
        fout.close()
        print(f"[{model}] wrote {n_new} template captions")
        continue

    from vlm import load_captioner   # imported lazily so the template baseline needs no torch
    t0 = time.time()
    cap = load_captioner(model, {**mcfg, "no_repeat_ngram_size": gen_cfg["no_repeat_ngram_size"]}, args.device)
    print(f"[{model}] loaded {mcfg['hf_id']} on {cap.device} ({cap.dtype}) in {time.time() - t0:.0f}s")

    # build the job list: (rec, obj, prompt_id, prompt, crop_box)
    jobs = []
    for rec in subset:
        if args.level in ("image", "both"):
            pid = mcfg["prompts"]["image"]
            jobs.append((rec, None, pid, fill_prompt(cfg["prompts"][pid]), None))
        if args.level in ("object", "both"):
            pid = mcfg["prompts"]["object"]
            for o in objects_to_caption(rec):
                jobs.append((rec, o, pid, fill_prompt(cfg["prompts"][pid], rec, o, tax), o["bbox"]))
    jobs = [j for j in jobs
            if f"{(j[1] or {}).get('object_id') or j[0]['image_id']}__{model}__{j[2]}__0" not in done]

    # group by level so a batch never mixes max_new_tokens
    by_level = defaultdict(list)
    for j in jobs:
        by_level["object" if j[1] else "image"].append(j)
    n_new = 0
    stats = defaultdict(list)
    for level, ljobs in by_level.items():
        bs = args.batch_size
        for i in tqdm(range(0, len(ljobs), bs), desc=f"{model}/{level}"):
            batch = ljobs[i:i + bs]
            imgs, boxes = [], []
            for rec, obj, _, _, bbox in batch:
                im = Image.open(ROOT / "data" / rec["file_name"]).convert("RGB")
                if obj:
                    im, box = crop_object(im, bbox, crop_cfg["context_pad"], crop_cfg["min_side"])
                    boxes.append(box)
                else:
                    boxes.append(None)
                imgs.append(im)
            t = time.time()
            outs = cap.generate(imgs, [j[3] for j in batch], gen_cfg[f"max_new_tokens_{level}"], n_cand,
                                gen_cfg["temperature"])
            per_item = (time.time() - t) / len(batch)
            stats[level].append(per_item)
            for (rec, obj, pid, prompt, _), box, cands in zip(batch, boxes, outs):
                for k, raw in enumerate(cands):
                    fout.write(json.dumps(make_row(model, mcfg, rec, obj, pid, prompt, raw, k, per_item, box)) + "\n")
                    n_new += 1
            fout.flush()
    fout.close()
    speed = {lv: f"{sum(v) / len(v):.2f}s/item" for lv, v in stats.items()}
    print(f"[{model}] wrote {n_new} captions  {speed}")
    del cap
