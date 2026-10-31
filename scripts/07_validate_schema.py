"""Validate every record in manifest_all.jsonl against schema/annotation_schema.json
and check that every image file exists and every box lies inside its image.
Week 2: also checks that every caption candidate is linked to the right image / object."""
import json

from jsonschema import Draft202012Validator

from common import PROCESSED, ROOT, read_jsonl

schema = json.load(open(ROOT / "schema" / "annotation_schema.json"))
validator = Draft202012Validator(schema)
recs = read_jsonl(PROCESSED / "manifest_all.jsonl")

errors = []
for r in recs:
    errors += [f"{r['image_id']}: {e.message}" for e in validator.iter_errors(r)]
    if not (ROOT / "data" / r["file_name"]).exists():
        errors.append(f"{r['image_id']}: missing image file")
    for o in r["objects"]:
        x, y, w, h = o["bbox"]
        if x < -1 or y < -1 or x + w > r["width"] + 1 or y + h > r["height"] + 1:
            errors.append(f"{o['object_id']}: bbox outside image")
        for c in o.get("caption_candidates", []):
            if not c["caption_id"].startswith(o["object_id"] + "__"):
                errors.append(f"{c['caption_id']}: object caption not linked to {o['object_id']}")
    for c in r["captions"]:
        if not c["caption_id"].startswith(r["image_id"] + "__"):
            errors.append(f"{c['caption_id']}: image caption not linked to {r['image_id']}")

ids = [r["image_id"] for r in recs]
if len(ids) != len(set(ids)):
    errors.append("duplicate image_id values")

n_caps = sum(len(r["captions"]) + sum(len(o.get("caption_candidates", [])) for o in r["objects"]) for r in recs)
print(f"validated {len(recs)} records ({n_caps} caption candidates), {len(errors)} problems")
for e in errors[:30]:
    print("  ", e)
