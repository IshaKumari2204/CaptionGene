"""Download a COCO val2017 subset containing waste-relevant classes (bottle, cup, wine glass).

Only instances_val2017.json is pulled out of the annotation zip (HTTP range requests).
We keep images where at least one target object is reasonably large (>= 1% of the
image) so captions have something meaningful to describe. Annotations for ALL
target-class objects in a kept image are retained (crowd regions are skipped).

Output: data/images/coco/*.jpg and data/interim/coco_records.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from remotezip import RemoteZip
from tqdm import tqdm

from common import IMAGES, INTERIM, RAW, SEED, process_image_bytes, scale_annotation, write_jsonl

ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TARGET = {"bottle", "cup", "wine glass"}
MIN_AREA_FRAC = 0.01

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400)
args = ap.parse_args()

p = RAW / "coco" / "instances_val2017.json"
if not p.exists():
    p.parent.mkdir(parents=True, exist_ok=True)
    with RemoteZip(ZIP_URL) as z:
        p.write_bytes(z.read("annotations/instances_val2017.json"))

d = json.load(open(p))
cats = {c["id"]: c["name"] for c in d["categories"] if c["name"] in TARGET}
imgs = {im["id"]: im for im in d["images"]}
anns = defaultdict(list)
for a in d["annotations"]:
    if a["category_id"] in cats and not a["iscrowd"]:
        anns[a["image_id"]].append(a)

eligible = [
    iid for iid, al in anns.items()
    if any(a["area"] / (imgs[iid]["width"] * imgs[iid]["height"]) >= MIN_AREA_FRAC for a in al)
]
# balance: alternate picks between images whose largest object is a bottle vs a cup/glass
by_main = defaultdict(list)
for iid in eligible:
    main = cats[max(anns[iid], key=lambda a: a["area"])["category_id"]]
    by_main[main].append(iid)
rng = random.Random(SEED)
for v in by_main.values():
    rng.shuffle(v)
chosen, i = [], 0
while len(chosen) < args.n and any(by_main.values()):
    key = sorted(by_main)[i % len(by_main)]
    if by_main[key]:
        chosen.append(by_main[key].pop())
    i += 1


def fetch(iid):
    im = imgs[iid]
    image_id = f"coco_{iid:012d}"
    out = IMAGES / "coco" / f"{image_id}.jpg"
    try:
        data = requests.get(im["coco_url"], timeout=60).content
        ow, oh, w, h, s = process_image_bytes(data, out, exif_fix=False)
    except Exception as e:  # noqa: BLE001
        return {"image_id": image_id, "error": repr(e)}
    return {
        "image_id": image_id,
        "source": "coco",
        "source_image_id": iid,
        "source_file_name": im["file_name"],
        "source_split": "val2017",
        "file_name": f"images/coco/{image_id}.jpg",
        "width": w, "height": h,
        "orig_width": im["width"], "orig_height": im["height"],
        "decoded_width": ow, "decoded_height": oh,
        "url": im["coco_url"],
        "license": "CC BY 4.0 (COCO, per-image Flickr license)",
        "source_scene_tags": [],
        "annotations": [
            {**scale_annotation(a, s), "source_category": cats[a["category_id"]]} for a in anns[iid]
        ],
    }


records = []
with ThreadPoolExecutor(16) as ex:
    futs = [ex.submit(fetch, iid) for iid in chosen]
    for f in tqdm(as_completed(futs), total=len(futs), desc="COCO"):
        records.append(f.result())
records.sort(key=lambda r: r["image_id"])
write_jsonl(INTERIM / "coco_records.jsonl", records)
print("eligible:", len(eligible), "ok:", sum("error" not in r for r in records),
      "errors:", sum("error" in r for r in records))
