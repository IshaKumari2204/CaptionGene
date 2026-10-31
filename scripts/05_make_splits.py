"""Create train / val / test splits (70 / 15 / 15).

* ZeroWaste keeps its official split (split by video, so no frame leakage).
* TACO + COCO are split with greedy multi-label iterative stratification so every
  waste category appears in all three splits in roughly 70/15/15 proportion.

Output: data/processed/{train,val,test}.jsonl, manifest_all.jsonl (with split filled in)
        and data/processed/objects.csv (one row per object, handy for spreadsheets)
"""
import csv
import random
from collections import Counter, defaultdict

from common import PROCESSED, SEED, read_jsonl, write_jsonl

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
rng = random.Random(SEED)

recs = read_jsonl(PROCESSED / "manifest_all.jsonl")
for r in recs:
    if r["source"] == "zerowaste":
        r["split"] = r["source_split"]

pool = [r for r in recs if r["split"] is None]
rng.shuffle(pool)
cat_freq = Counter(c for r in pool for c in set(r["scene"]["category_counts"]))
desired = {s: {c: n * p for c, n in cat_freq.items()} for s, p in RATIOS.items()}
desired_size = {s: len(pool) * p for s, p in RATIOS.items()}

# rarest-category-first assignment
pool.sort(key=lambda r: min(cat_freq[c] for c in r["scene"]["category_counts"]))
for r in pool:
    cats = set(r["scene"]["category_counts"])
    rarest = min(cats, key=lambda c: cat_freq[c])
    best = max(RATIOS, key=lambda s: (desired[s][rarest], desired_size[s]))
    r["split"] = best
    desired_size[best] -= 1
    for c in cats:
        desired[best][c] -= 1

write_jsonl(PROCESSED / "manifest_all.jsonl", recs)
by_split = defaultdict(list)
for r in recs:
    by_split[r["split"]].append(r)
for s in RATIOS:
    write_jsonl(PROCESSED / f"{s}.jsonl", by_split[s])

with open(PROCESSED / "objects.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "split", "source", "file_name", "object_id", "source_category",
                "category", "material", "material_detail", "sorting_action", "x", "y", "w", "h",
                "region", "size", "flags", "environment", "clutter_level"])
    for r in recs:
        for o in r["objects"]:
            w.writerow([r["image_id"], r["split"], r["source"], r["file_name"], o["object_id"],
                        o["source_category"], o["category"], o["material"], o["material_detail"],
                        o["sorting_action"], *o["bbox"], o["location"]["region"],
                        o["location"]["size"], "|".join(o["flags"]),
                        "|".join(r["scene"]["environment"]), r["scene"]["clutter_level"]])

for s in RATIOS:
    print(s, len(by_split[s]), dict(Counter(r["source"] for r in by_split[s])))
