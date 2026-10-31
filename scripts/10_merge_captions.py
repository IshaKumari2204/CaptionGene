"""Merge raw caption candidates (data/captions/*.jsonl) and human review events
(data/reviews/reviews.jsonl, written by the Streamlit app) into the annotation records.

  * image-level candidates  -> record["captions"]
  * object-level candidates -> record["objects"][i]["caption_candidates"]   (linked by object_id
                               to the object's bbox + segmentation mask)
  * every candidate gets auto_checks (material / category mentions, unsupported objects, brands)

The merge is rebuilt from scratch each time, so it is safe to re-run (also after re-running
scripts 04/05, which reset the records).

Output: data/processed/manifest_all.jsonl + train/val/test.jsonl (updated in place)
        data/processed/captions.csv (one row per candidate, for spreadsheets)
"""
import csv

from captionlib import load_generations, load_review_events, merge_records
from common import PROCESSED, read_jsonl, write_jsonl

records = read_jsonl(PROCESSED / "manifest_all.jsonl")
gens = load_generations()
events = load_review_events()
merge_records(records, gens, events)

write_jsonl(PROCESSED / "manifest_all.jsonl", records)
for split in ("train", "val", "test"):
    write_jsonl(PROCESSED / f"{split}.jsonl", [r for r in records if r["split"] == split])

with open(PROCESSED / "captions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["caption_id", "image_id", "object_id", "level", "split", "source", "model", "prompt_id",
                "text", "clip_score", "status", "review_score", "category", "material", "proposed_condition",
                "mentions_material", "material_match", "unsupported_categories", "brand_or_text"])
    for r in records:
        rows = [(c, None) for c in r["captions"]]
        rows += [(c, o) for o in r["objects"] for c in o["caption_candidates"]]
        for c, o in rows:
            a = c["auto_checks"]
            w.writerow([c["caption_id"], r["image_id"], o["object_id"] if o else "", "object" if o else "image",
                        r["split"], r["source"], c["model"], c.get("prompt_id"), c["text"], c["clip_score"],
                        c["status"], c["review_score"], o["category"] if o else "", o["material"] if o else "",
                        "|".join(c.get("proposed", {}).get("condition", [])),
                        a.get("mentions_material", bool(a["mentioned_materials"])), a.get("material_match", ""),
                        "|".join(a.get("unsupported_categories", [])), a["brand_or_text"]])

n_img = sum(len(r["captions"]) for r in records)
n_obj = sum(len(o["caption_candidates"]) for r in records for o in r["objects"])
n_imgs_cap = sum(1 for r in records if r["captions"])
print(f"merged {len(gens)} generations + {len(events)} review events -> "
      f"{n_img} image captions on {n_imgs_cap} images, {n_obj} object captions")
