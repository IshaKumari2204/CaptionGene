"""Download all 1,500 TACO images, fix EXIF orientation, resize, and scale annotations.

Output: data/images/taco/*.jpg and data/interim/taco_records.jsonl
"""
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

from common import IMAGES, INTERIM, RAW, process_image_bytes, scale_annotation, write_jsonl

ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
ann_path = RAW / "taco" / "annotations.json"
if not ann_path.exists():
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_bytes(requests.get(ANN_URL, timeout=60).content)

d = json.load(open(ann_path))
cats = {c["id"]: c["name"] for c in d["categories"]}
scene_cats = {c["id"]: c["name"] for c in d["scene_categories"]}
anns = defaultdict(list)
for a in d["annotations"]:
    anns[a["image_id"]].append(a)
# Some images have several scene annotations; keep tags chosen by >= half of them.
# Background id 7 is not listed in scene_categories, so it is mapped to "unknown".
_votes, _n = defaultdict(Counter), Counter()
for s in d["scene_annotations"]:
    _n[s["image_id"]] += 1
    _votes[s["image_id"]].update({scene_cats.get(b, "unknown") for b in s["background_ids"]})
scenes = {i: {t for t, c in v.items() if c >= _n[i] / 2} for i, v in _votes.items()}


def download(url, deadline=90):
    """GET with a hard total-time limit (some image hosts trickle bytes forever)."""
    start, buf = time.time(), bytearray()
    with requests.get(url, timeout=(10, 30), stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 16):
            buf += chunk
            if time.time() - start > deadline:
                raise TimeoutError(f"download exceeded {deadline}s")
    return bytes(buf)


def fetch(img):
    image_id = f"taco_{img['id']:05d}"
    out = IMAGES / "taco" / f"{image_id}.jpg"
    try:
        if out.exists():  # resume: reuse an already processed image
            w, h = Image.open(out).size
            ow, oh = img["width"], img["height"]
            if abs(w / h - ow / oh) > 0.02:  # aspect differs -> orientation mismatch
                ow, oh = oh, ow
            s = w / ow
        else:
            data = download(img["flickr_url"])
            ow, oh, w, h, s = process_image_bytes(data, out, exif_fix=True)
    except Exception as e:  # noqa: BLE001
        return {"image_id": image_id, "error": repr(e)}
    return {
        "image_id": image_id,
        "source": "taco",
        "source_image_id": img["id"],
        "source_file_name": img["file_name"],
        "source_split": None,
        "file_name": f"images/taco/{image_id}.jpg",
        "width": w, "height": h,
        "orig_width": img["width"], "orig_height": img["height"],
        "decoded_width": ow, "decoded_height": oh,
        "url": img["flickr_url"],
        "license": "CC BY 4.0 (TACO)",
        "source_scene_tags": sorted(scenes.get(img["id"], [])),
        "annotations": [
            {**scale_annotation(a, s), "source_category": cats[a["category_id"]]}
            for a in anns[img["id"]]
        ],
    }


records = []
with ThreadPoolExecutor(16) as ex:
    futs = [ex.submit(fetch, img) for img in d["images"]]
    for f in tqdm(as_completed(futs), total=len(futs), desc="TACO"):
        records.append(f.result())
records.sort(key=lambda r: r["image_id"])
write_jsonl(INTERIM / "taco_records.jsonl", records)
print("ok:", sum("error" not in r for r in records), "errors:", sum("error" in r for r in records))
