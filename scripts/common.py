"""Shared paths and helpers for the CaptionGene data pipeline."""
from __future__ import annotations

import io
import json
from pathlib import Path

import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
IMAGES = ROOT / "data" / "images"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
TAXONOMY_PATH = ROOT / "configs" / "taxonomy.yaml"

MAX_SIDE = 1280      # processed images are resized so the longer side is <= 1280 px
JPEG_QUALITY = 90
SEED = 42

for _d in (RAW, INTERIM, IMAGES, PROCESSED, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def process_image_bytes(data: bytes, out_path: Path, exif_fix: bool = True):
    """Decode, fix EXIF orientation, resize to MAX_SIDE and save as JPEG.

    Returns (orig_w, orig_h, new_w, new_h, scale) where orig_* is the size
    after EXIF transpose (the frame the source annotations live in).
    """
    img = Image.open(io.BytesIO(data))
    img.load()
    if exif_fix:
        img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    ow, oh = img.size
    scale = min(1.0, MAX_SIDE / max(ow, oh))
    if scale < 1.0:
        img = img.resize((round(ow * scale), round(oh * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=JPEG_QUALITY)
    return ow, oh, img.size[0], img.size[1], scale


def scale_annotation(ann: dict, scale: float) -> dict:
    """Scale a COCO-style annotation (bbox + polygon segmentation) by `scale`."""
    out = dict(ann)
    out["bbox"] = [round(v * scale, 2) for v in ann["bbox"]]
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        out["segmentation"] = [[round(v * scale, 2) for v in poly] for poly in seg]
    else:
        out["segmentation"] = None  # RLE masks are dropped (only used for COCO crowd regions)
    out["area"] = round(ann.get("area", 0) * scale * scale, 2)
    return out


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
