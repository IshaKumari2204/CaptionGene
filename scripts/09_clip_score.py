"""Compute reference-free CLIPScore for every generated caption.

CLIPScore (Hessel et al., 2021) = 2.5 * max(cos(E_image, E_text), 0) with CLIP ViT-B/32 and the
text prefixed by "A photo depicts". Range is 0-2.5; human-quality captions usually score 0.7-0.9.
Object captions are scored against the same padded crop the model saw (crop_box).

The score is written back into data/captions/<model>.jsonl (field clip_score).

  python 09_clip_score.py                 # all models, only rows without a score
  python 09_clip_score.py --recompute
"""
import argparse

from PIL import Image
from tqdm import tqdm

from captionlib import CAPTIONS
from common import PROCESSED, ROOT, read_jsonl, write_jsonl
from vlm import ClipScorer

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--recompute", action="store_true")
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

scorer = ClipScorer()

_img_cache = {}
file_of = {r["image_id"]: r["file_name"] for r in read_jsonl(PROCESSED / "manifest_all.jsonl")}


def load_view(row):
    key = (row["image_id"], tuple(row["crop_box"]) if row.get("crop_box") else None)
    if key not in _img_cache:
        if len(_img_cache) > 256:
            _img_cache.clear()
        im = Image.open(ROOT / "data" / file_of[row["image_id"]]).convert("RGB")
        _img_cache[key] = im.crop(row["crop_box"]) if row.get("crop_box") else im
    return _img_cache[key]


for path in sorted(CAPTIONS.glob("*.jsonl")):
    if args.models and path.stem not in args.models:
        continue
    rows = read_jsonl(path)
    todo = [r for r in rows if r["text"].strip() and (args.recompute or r.get("clip_score") is None)]
    for i in tqdm(range(0, len(todo), args.batch_size), desc=path.stem):
        batch = todo[i:i + args.batch_size]
        for r, s in zip(batch, scorer.score([load_view(r) for r in batch], [r["text"] for r in batch])):
            r["clip_score"] = s
    write_jsonl(path, rows)
    scored = [r["clip_score"] for r in rows if r.get("clip_score") is not None]
    print(f"{path.stem}: {len(todo)} newly scored, mean CLIPScore {sum(scored) / max(1, len(scored)):.3f}")
