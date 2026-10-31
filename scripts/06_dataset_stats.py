"""Compute dataset statistics and render a sample grid with boxes for visual sanity checks.

Output: data/reports/stats.json, data/reports/stats.md, data/reports/sample_grid.jpg
"""
import json
import random
from collections import Counter

from PIL import Image, ImageDraw

from common import PROCESSED, REPORTS, ROOT, SEED, read_jsonl

recs = read_jsonl(PROCESSED / "manifest_all.jsonl")
objs = [(r, o) for r in recs for o in r["objects"]]
splits = ["train", "val", "test"]


def table(counter_by_split, title, key_order=None):
    keys = key_order or sorted({k for c in counter_by_split.values() for k in c},
                               key=lambda k: -sum(c.get(k, 0) for c in counter_by_split.values()))
    lines = [f"| {title} | " + " | ".join(splits) + " | total |",
             "|---|" + "---:|" * (len(splits) + 1)]
    for k in keys:
        vals = [counter_by_split[s].get(k, 0) for s in splits]
        lines.append(f"| {k} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")
    return "\n".join(lines)


def by_split(fn, unit="obj"):
    out = {s: Counter() for s in splits}
    if unit == "obj":
        for r, o in objs:
            out[r["split"]][fn(r, o)] += 1
    else:
        for r in recs:
            for v in fn(r):
                out[r["split"]][v] += 1
    return out


stats = {
    "images": len(recs),
    "objects": len(objs),
    "images_per_split": dict(Counter(r["split"] for r in recs)),
    "images_per_source": dict(Counter(r["source"] for r in recs)),
    "objects_per_category": dict(Counter(o["category"] for _, o in objs).most_common()),
    "objects_per_material": dict(Counter(o["material"] for _, o in objs).most_common()),
    "objects_per_sorting_action": dict(Counter(o["sorting_action"] for _, o in objs).most_common()),
    "images_per_clutter_level": dict(Counter(r["scene"]["clutter_level"] for r in recs)),
    "images_per_environment": dict(Counter(e for r in recs for e in r["scene"]["environment"])),
    "object_size_buckets": dict(Counter(o["location"]["size"] for _, o in objs)),
    "object_flags": dict(Counter(f for _, o in objs for f in o["flags"])),
    "image_quality_flags": dict(Counter(f for r in recs for f in r["quality"]["flags"])),
    "mean_objects_per_image": round(len(objs) / len(recs), 2),
    "max_objects_per_image": max(r["scene"]["object_count"] for r in recs),
}
json.dump(stats, open(REPORTS / "stats.json", "w"), indent=2)

md = [
    "# CaptionGene dataset statistics (Week 1, pre-captioning)", "",
    f"- **Images:** {stats['images']}  |  **Objects:** {stats['objects']}  |  "
    f"**Mean objects/image:** {stats['mean_objects_per_image']}  |  "
    f"**Max objects/image:** {stats['max_objects_per_image']}", "",
    "## Images per source and split", "",
    table(by_split(lambda r: [r["source"]], "img"), "source"), "",
    "## Objects per waste category", "",
    table(by_split(lambda r, o: o["category"]), "category"), "",
    "## Objects per material", "",
    table(by_split(lambda r, o: o["material"]), "material"), "",
    "## Objects per default sorting action", "",
    table(by_split(lambda r, o: o["sorting_action"]), "sorting_action"), "",
    "## Images per clutter level", "",
    table(by_split(lambda r: [r["scene"]["clutter_level"]], "img"), "clutter",
          ["single", "sparse", "moderate", "dense"]), "",
    "## Images per scene environment (an image may have several)", "",
    table(by_split(lambda r: r["scene"]["environment"], "img"), "environment"), "",
    "## Object size and flags", "",
    table(by_split(lambda r, o: o["location"]["size"]), "size", ["small", "medium", "large"]), "",
    f"Object flags: {stats['object_flags']}  ", "",
    f"Image quality flags: {stats['image_quality_flags']}", "",
]
(REPORTS / "stats.md").write_text("\n".join(md))

# sample grid: 4 images per source with boxes drawn
COLORS = {"plastic": "#1f77b4", "metal": "#7f7f7f", "glass": "#17becf", "paper_cardboard": "#8c564b",
          "organic": "#2ca02c", "composite": "#9467bd", "textile_rubber": "#e377c2",
          "e_waste": "#d62728", "unknown": "#ff7f0e"}
rng = random.Random(SEED)
tiles = []
for src in ("taco", "zerowaste", "coco"):
    pool = [r for r in recs if r["source"] == src and r["scene"]["object_count"] >= 2]
    for r in rng.sample(pool, min(4, len(pool))):
        im = Image.open(ROOT / "data" / r["file_name"]).convert("RGB")
        dr = ImageDraw.Draw(im)
        for o in r["objects"]:
            x, y, w, h = o["bbox"]
            dr.rectangle([x, y, x + w, y + h], outline=COLORS[o["material"]], width=4)
            dr.text((x + 4, y + 4), f"{o['category']}/{o['material']}", fill=COLORS[o["material"]])
        im.thumbnail((480, 360))
        tile = Image.new("RGB", (480, 380), "white")
        tile.paste(im, ((480 - im.width) // 2, 0))
        ImageDraw.Draw(tile).text((6, 364), f"{r['image_id']}  [{r['split']}]", fill="black")
        tiles.append(tile)
grid = Image.new("RGB", (480 * 4, 380 * 3), "white")
for i, t in enumerate(tiles):
    grid.paste(t, ((i % 4) * 480, (i // 4) * 380))
grid.save(REPORTS / "sample_grid.jpg", quality=88)
print(json.dumps(stats, indent=2))
