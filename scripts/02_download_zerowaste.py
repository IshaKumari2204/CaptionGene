"""Download a diverse subset of ZeroWaste-f (the full zip is 7.5 GB).

Uses HTTP range requests (remotezip) so only the selected frames are fetched.
Frames come from conveyor-belt videos, so consecutive frames are near-duplicates.
We keep ZeroWaste's official video-level splits (no leakage) and sample frames
evenly spaced in time within each video.

Output: data/images/zerowaste/*.jpg and data/interim/zerowaste_records.jsonl
"""
import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from remotezip import RemoteZip
from tqdm import tqdm

from common import IMAGES, INTERIM, RAW, process_image_bytes, scale_annotation, write_jsonl

URL = "https://zenodo.org/records/6412647/files/zerowaste-f-final.zip?download=1"
PREFIX = "splits_final_deblurred"

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=420)
ap.add_argument("--val", type=int, default=90)
ap.add_argument("--test", type=int, default=90)
args = ap.parse_args()
quota = {"train": args.train, "val": args.val, "test": args.test}

(RAW / "zerowaste").mkdir(parents=True, exist_ok=True)
with RemoteZip(URL) as z:
    for split in quota:
        p = RAW / "zerowaste" / f"labels_{split}.json"
        if not p.exists():
            p.write_bytes(z.read(f"{PREFIX}/{split}/labels.json"))


def select(images, anns, n):
    """Pick n frames: spread across videos, evenly spaced in time, non-empty only."""
    by_video = defaultdict(list)
    for im in images:
        if anns[im["id"]]:
            by_video[im["file_name"].split("_")[0]].append(im)
    total = sum(len(v) for v in by_video.values())
    chosen = []
    for vid, ims in sorted(by_video.items()):
        ims.sort(key=lambda im: im["file_name"])
        k = max(1, round(n * len(ims) / total))
        step = len(ims) / k
        chosen += [ims[int(i * step)] for i in range(k)]
    return chosen[:n]


_local = threading.local()


def fetch(split, im, cats, anns):
    image_id = f"zerowaste_{split}_{im['id']:05d}"
    out = IMAGES / "zerowaste" / f"{image_id}.jpg"
    try:
        if not hasattr(_local, "zip"):  # one remote handle per worker thread
            _local.zip = RemoteZip(URL)
        data = _local.zip.read(f"{PREFIX}/{split}/data/{im['file_name']}")
        ow, oh, w, h, s = process_image_bytes(data, out, exif_fix=False)
    except Exception as e:  # noqa: BLE001
        return {"image_id": image_id, "error": repr(e)}
    return {
        "image_id": image_id,
        "source": "zerowaste",
        "source_image_id": im["id"],
        "source_file_name": im["file_name"],
        "source_split": split,
        "video_id": im["file_name"].split("_")[0],
        "file_name": f"images/zerowaste/{image_id}.jpg",
        "width": w, "height": h,
        "orig_width": im["width"], "orig_height": im["height"],
        "decoded_width": ow, "decoded_height": oh,
        "url": URL,
        "license": "CC BY-NC 4.0 (ZeroWaste)",
        "source_scene_tags": ["conveyor_belt"],
        "annotations": [
            {**scale_annotation(a, s), "source_category": cats[a["category_id"]]}
            for a in anns[im["id"]]
        ],
    }


records = []
for split, n in quota.items():
    d = json.load(open(RAW / "zerowaste" / f"labels_{split}.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    chosen = select(d["images"], anns, n)
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(fetch, split, im, cats, anns) for im in chosen]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"ZeroWaste {split}"):
            records.append(f.result())

records.sort(key=lambda r: r["image_id"])
write_jsonl(INTERIM / "zerowaste_records.jsonl", records)
print("ok:", sum("error" not in r for r in records), "errors:", sum("error" in r for r in records))
