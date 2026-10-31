"""Clean all sources and convert them into the unified CaptionGene annotation schema.

Cleaning steps (each one is logged in data/reports/cleaning_report.json):
  1. drop images that failed to download / decode
  2. drop images below the minimum resolution
  3. drop images whose decoded size does not match the source annotation frame
     (e.g. EXIF rotation problems) - boxes would be misaligned
  4. drop annotations with degenerate boxes (w or h < 2 px)
  5. drop images left with no valid objects
  6. drop exact duplicates (MD5) and near-duplicates (perceptual hash, Hamming <= 4)
  7. flag (not drop) blurry / very dark / very bright images for later review

Output: data/processed/manifest_all.jsonl
"""
import hashlib
import json
from collections import Counter, defaultdict

import imagehash
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import INTERIM, PROCESSED, REPORTS, ROOT, load_taxonomy, read_jsonl, write_jsonl

MIN_SIDE = 224
PHASH_MAX_DIST = 4
TINY_AREA_FRAC = 0.0005     # objects smaller than 0.05% of the image are flagged as tiny
OCCLUSION_IOA = 0.30        # >30% of a box covered by other boxes -> possibly occluded

tax = load_taxonomy()
maps = tax["source_maps"]
log = Counter()
dropped = defaultdict(list)


def region_of(bbox, w, h):
    cx, cy = (bbox[0] + bbox[2] / 2) / w, (bbox[1] + bbox[3] / 2) / h
    col = ["left", "center", "right"][min(2, int(cx * 3))]
    row = ["top", "middle", "bottom"][min(2, int(cy * 3))]
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


def size_of(frac):
    return "small" if frac < 0.01 else "medium" if frac < 0.10 else "large"


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def clutter_level(n):
    for name, (lo, hi) in tax["clutter_levels"].items():
        if lo <= n <= hi:
            return name
    return "unknown"


def image_quality(path):
    img = Image.open(path).convert("L")
    small = img.resize((256, round(256 * img.height / img.width)))
    arr = np.asarray(small, dtype=np.float32)
    lap = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return {"brightness": round(float(arr.mean()), 1), "sharpness": round(float(lap.var()), 1)}


def convert(rec):
    src = rec["source"]
    w, h = rec["width"], rec["height"]
    objs = []
    for i, a in enumerate(rec["annotations"]):
        x, y, bw, bh = a["bbox"]
        # clip boxes that slightly exceed the image border (annotation noise)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        x, y = max(0.0, x), max(0.0, y)
        bw, bh = x2 - x, y2 - y
        a = {**a, "bbox": [x, y, bw, bh]}
        if bw < 2 or bh < 2:
            log["annotations_dropped_degenerate_bbox"] += 1
            continue
        cat, mat, mat_detail, action = maps[src][a["source_category"]]
        frac = (bw * bh) / (w * h)
        objs.append({
            "object_id": f"{rec['image_id']}_o{i:03d}",
            "source_category": a["source_category"],
            "category": cat,
            "material": mat,
            "material_detail": mat_detail,
            "condition": ["broken"] if a["source_category"] == "Broken glass" else [],
            "sorting_action": action,
            "bbox": [round(x, 1), round(y, 1), round(bw, 1), round(bh, 1)],
            "bbox_norm": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)],
            "segmentation": a.get("segmentation"),
            "area": a["area"],
            "location": {"region": region_of(a["bbox"], w, h), "size": size_of(frac)},
            "flags": ["tiny"] if frac < TINY_AREA_FRAC else [],
            "caption": None,
            "instruction": None,
        })
    # occlusion hint from box overlaps
    for o in objs:
        own = o["bbox"][2] * o["bbox"][3]
        cov = sum(inter(o["bbox"], p["bbox"]) for p in objs if p is not o)
        if own and cov / own > OCCLUSION_IOA:
            o["flags"].append("possibly_occluded")

    tags = rec["source_scene_tags"]
    if src == "taco":
        env = sorted({maps["taco_scene"][t] for t in tags}) or ["unknown"]
    elif src == "zerowaste":
        env = ["conveyor_belt"]
    else:
        env = ["unknown"]   # COCO scenes vary; filled by VLM/human later
    return {
        "image_id": rec["image_id"],
        "source": src,
        "source_image_id": rec["source_image_id"],
        "source_file_name": rec["source_file_name"],
        "source_split": rec["source_split"],
        "video_id": rec.get("video_id"),
        "file_name": rec["file_name"],
        "width": w, "height": h,
        "orig_width": rec["orig_width"], "orig_height": rec["orig_height"],
        "license": rec["license"],
        "url": rec["url"],
        "split": None,
        "scene": {
            "environment": env,
            "clutter_level": clutter_level(len(objs)),
            "object_count": len(objs),
            "category_counts": dict(Counter(o["category"] for o in objs)),
            "is_discarded": "unknown" if src == "coco" else "yes",
            "description": None,
        },
        "objects": objs,
        "captions": [],
        "review": {"status": "unreviewed", "reviewer": None, "notes": None, "updated_at": None},
    }


records = []
for src in ("taco", "zerowaste", "coco"):
    rows = read_jsonl(INTERIM / f"{src}_records.jsonl")
    log[f"{src}_downloaded_attempted"] = len(rows)
    for r in rows:
        if "error" in r:
            dropped["download_or_decode_error"].append(r["image_id"])
            continue
        if min(r["width"], r["height"]) < MIN_SIDE:
            dropped["below_min_resolution"].append(r["image_id"])
            continue
        if (r["decoded_width"], r["decoded_height"]) != (r["orig_width"], r["orig_height"]):
            dropped["annotation_frame_mismatch"].append(r["image_id"])
            continue
        rec = convert(r)
        if not rec["objects"]:
            dropped["no_valid_objects"].append(r["image_id"])
            continue
        records.append(rec)

# duplicates
seen_md5, kept, hashes = {}, [], []
for rec in tqdm(records, desc="dedup + quality"):
    p = ROOT / "data" / rec["file_name"]
    if not p.exists():  # re-run: file was moved to data/dropped/ by a previous run
        alt = ROOT / "data" / "dropped" / rec["source"] / p.name
        alt.rename(p)
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    if md5 in seen_md5:
        dropped["exact_duplicate"].append(rec["image_id"])
        continue
    seen_md5[md5] = rec["image_id"]
    ph = imagehash.phash(Image.open(p))
    near = next((hid for hh, hid in hashes if ph - hh <= PHASH_MAX_DIST), None)
    if near:
        dropped["near_duplicate"].append(f"{rec['image_id']} ~ {near}")
        continue
    hashes.append((ph, rec["image_id"]))
    rec["phash"] = str(ph)
    q = image_quality(p)
    rec["quality"] = {**q, "flags": []}
    if q["sharpness"] < 50:
        rec["quality"]["flags"].append("blurry")
    if q["brightness"] < 40:
        rec["quality"]["flags"].append("too_dark")
    if q["brightness"] > 230:
        rec["quality"]["flags"].append("too_bright")
    kept.append(rec)

# move files of dropped images to data/dropped/ so data/images only holds the clean set
kept_files = {r["file_name"] for r in kept}
for src in ("taco", "zerowaste", "coco"):
    for f in (ROOT / "data" / "images" / src).glob("*.jpg"):
        if f"images/{src}/{f.name}" not in kept_files:
            dest = ROOT / "data" / "dropped" / src / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.rename(dest)

write_jsonl(PROCESSED / "manifest_all.jsonl", kept)
report = {
    "counts": dict(log),
    "kept_images": len(kept),
    "kept_by_source": dict(Counter(r["source"] for r in kept)),
    "dropped_counts": {k: len(v) for k, v in dropped.items()},
    "dropped": dropped,
    "flagged_quality": dict(Counter(f for r in kept for f in r["quality"]["flags"])),
    "params": {"MIN_SIDE": MIN_SIDE, "PHASH_MAX_DIST": PHASH_MAX_DIST,
               "TINY_AREA_FRAC": TINY_AREA_FRAC, "OCCLUSION_IOA": OCCLUSION_IOA},
}
json.dump(report, open(REPORTS / "cleaning_report.json", "w"), indent=2)
print(json.dumps({k: v for k, v in report.items() if k != "dropped"}, indent=2))
