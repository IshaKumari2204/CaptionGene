# Week 1 Updates: Project Setup & Dataset Preparation

**Project:** Vision-Language Caption Generation for Waste Images (CaptionGene)
**Week 1 goal (from plan):** Study TACO and ZeroWaste; finalize waste categories, annotation format, and project scope. Collect, clean, and organize diverse waste images into train/val/test splits. Define the annotation schema (object category, material, condition, location, sorting action).

**Status: ✅ All Week 1 tasks completed.**

---

## 1. Summary (TL;DR)

| Item | Result |
|---|---|
| Sources studied and integrated | TACO, ZeroWaste-f, COCO (bottle / cup / wine glass) |
| Clean images in dataset | **2,410** (TACO 1,493 · ZeroWaste 597 · COCO 320) |
| Annotated objects (boxes + masks) | **9,391** (mean 3.9 per image, max 90) |
| Unified taxonomy | 15 object categories · 9 materials · 13 conditions · 8 sorting actions · 10 scene environments |
| Annotation schema | JSON Schema v1.0 (`schema/annotation_schema.json`); all 2,410 records validate with **0 errors** |
| Splits | train 1,712 / val 355 / test 343 (≈ 71 / 15 / 14 %) |
| Reproducible pipeline | 7 Python scripts, fixed seed 42, one command per step |

---

## 2. Dataset study

### 2.1 TACO (Trash Annotations in Context)
- **What:** 1,500 photos of litter in the wild (beaches, parks, streets, indoors), taken with phones and hosted on Flickr.
- **Labels:** 4,784 objects with **instance segmentation masks + boxes**, 60 fine categories in 28 super-categories (e.g. *Clear plastic bottle*, *Drink can*, *Crisp packet*, *Cigarette*). Also has image-level **scene/background tags** (Vegetation, Pavement, Sand, Water, …).
- **Strengths:** very fine-grained object types, real "in-context" clutter, partial occlusion and damaged objects. Its labels mix object type and material (e.g. "Glass bottle"), which makes our material mapping easy.
- **Limitations:** small; long-tailed (e.g. only 2 batteries, 8 food-waste objects); many objects are tiny; full-resolution images are large (~3 MB each, up to 4000+ px); EXIF orientation must be applied or the masks are misaligned.
- **License:** CC BY 4.0.

### 2.2 ZeroWaste (ZeroWaste-f, CVPR 2022)
- **What:** frames from videos of a real **material recovery facility (MRF) conveyor belt**: very dense, deformed, overlapping waste.
- **Labels:** 4,503 frames (train 3,002 / val 572 / test 929, split **by video**), 26,766 polygon annotations in only **4 material classes**: cardboard (66%), soft plastic (25%), rigid plastic (6%), metal (1%).
- **Strengths:** extreme clutter and deformation, which is exactly the hard case for VLM captions.
- **Limitations:** labels give material only (no object type); paper and other items are left unlabeled as "background"; consecutive frames are near-duplicates; the full archive is 7.5 GB.
- **License:** CC BY-NC 4.0 (non-commercial).

### 2.3 COCO 2017 (relevant classes)
- COCO has no "can" or "trash" classes. The waste-relevant ones are **bottle, cup, wine glass**.
- **Role in our dataset:** a **domain-contrast / hard-negative set**. Mostly the objects are *in use*, not discarded (verified visually, see sample grid). This tests whether a VLM wrongly calls every bottle "waste". So these records get `is_discarded: "unknown"`, and their material and sorting action are set to `needs_review`.

### 2.4 Comparison

| | TACO | ZeroWaste-f | COCO subset |
|---|---|---|---|
| Scene type | litter in the wild | industrial conveyor | everyday scenes |
| Label granularity | 60 object types | 4 materials | 3 object types |
| Masks | ✅ | ✅ | ✅ |
| Clutter | low–medium | very high | low–medium |
| What it adds to CaptionGene | object type + material + context | dense clutter + deformation | non-waste contrast |

---

## 3. Scope decisions (finalized)

1. **Image pool:** all of TACO + a **597-frame** subset of ZeroWaste-f + **320** COCO val2017 images. This is about 2.4k images: large enough to be diverse, small enough to caption with 2–3 VLMs and still human-verify in Weeks 2–3.
2. **ZeroWaste subset:** I downloaded only the selected frames with **HTTP range requests** (the full 7.5 GB zip was never downloaded). Frames are **sampled evenly in time within each video** to avoid near-duplicates, and **ZeroWaste's official video-level split is kept** so frames from one video never leak across train/test.
3. **COCO filter:** keep val2017 images where at least one bottle / cup / wine glass covers ≥ 1% of the image. Crowd regions are skipped. Selection is balanced between "bottle-dominant" and "cup/glass-dominant" images.
4. **Image format:** EXIF-corrected, resized so the longest side ≤ 1280 px, JPEG q90. Boxes and polygons are rescaled to match. Original size and source URL are kept for traceability.
5. **"In-the-wild images from search":** deferred. They add licensing risk and have no ground-truth boxes. The schema already supports `source: "wild"` if we add a small set later.
6. **Captioning granularity:** both **image-level captions** and **object-level captions/instructions** linked to a box/mask via `object_id`.

---

## 4. Unified taxonomy (`configs/taxonomy.yaml`)

Each object has four independent fields, so captions can be composed as **"<condition> <material> <category>"** → *"crushed plastic bottle"*.

**Object categories (15):** `bottle_jar, can, carton_box, cup, bag_wrapper, container_packaging, cap_lid, paper, straw_utensil, cigarette, food_waste, glass_fragment, hazardous, other_litter, unspecified`.
`unspecified` is used for ZeroWaste rigid/soft plastic and metal, whose object type is unknown. The VLM proposes a type in Week 2 and a human verifies it in Week 3.

**Materials (9):** `plastic, metal, glass, paper_cardboard, organic, composite, textile_rubber, e_waste, unknown`.

**Conditions (13, multi-label):** `intact, crushed, dented, crumpled, torn, broken, open, soiled, wet, burnt, faded, partially_occluded, heavily_occluded`. These are empty for now and get filled in Weeks 2–3. Exception: TACO "Broken glass" is pre-filled with `broken`.

**Sorting actions (8):** `recycle_plastic, recycle_metal, recycle_paper, recycle_glass, compost, landfill, hazardous_dropoff, needs_review`. Each source label has a default action, plus **condition-based overrides** (e.g. a *soiled* paper item goes to landfill).

**Location:** box + normalized box + 3×3 grid region (e.g. `bottom-left`) + size bucket (small < 1%, medium < 10%, large ≥ 10% of the image).

**Scene:** environment tags (TACO's scene labels are mapped; ZeroWaste = `conveyor_belt`), clutter level (single / sparse / moderate / dense, from the object count), per-category counts, and `is_discarded`.

All **60 TACO classes, 4 ZeroWaste classes and 3 COCO classes are mapped explicitly**. Examples:

| Source label | category | material | detail | default action |
|---|---|---|---|---|
| TACO "Clear plastic bottle" | bottle_jar | plastic | PET | recycle_plastic |
| TACO "Drink carton" | carton_box | composite | tetra_pak | recycle_paper |
| TACO "Crisp packet" | bag_wrapper | composite | metallised | landfill |
| TACO "Pizza box" | carton_box | paper_cardboard | – | recycle_paper (→ landfill if soiled) |
| ZeroWaste "soft_plastic" | unspecified | plastic | film | landfill |
| COCO "bottle" | bottle_jar | unknown | – | needs_review |

---

## 5. Annotation schema (`schema/annotation_schema.json`)

There is one JSON record per image. It is stored as JSONL (`data/processed/*.jsonl`), and a flat per-object CSV (`objects.csv`) is provided for spreadsheets. Abbreviated example of a real record:

```json
{
  "image_id": "taco_00001", "source": "taco", "split": "val",
  "file_name": "images/taco/taco_00001.jpg", "width": 960, "height": 1280,
  "orig_width": 1537, "orig_height": 2049, "license": "CC BY 4.0 (TACO)",
  "scene": {"environment": ["indoor_manmade"], "clutter_level": "sparse",
            "object_count": 2, "category_counts": {"carton_box": 2},
            "is_discarded": "yes", "description": null},
  "objects": [{
      "object_id": "taco_00001_o000", "source_category": "Meal carton",
      "category": "carton_box", "material": "paper_cardboard", "material_detail": null,
      "condition": [], "sorting_action": "recycle_paper",
      "bbox": [0.6, 285.4, 892.5, 948.8], "bbox_norm": [0.0006, 0.223, 0.9297, 0.7412],
      "segmentation": [[...]], "location": {"region": "center", "size": "large"},
      "flags": [], "caption": null, "instruction": null }],
  "captions": [],
  "review": {"status": "unreviewed", "reviewer": null, "notes": null, "updated_at": null}
}
```

- `captions[]` will hold VLM candidates (`model`, `prompt`, `clip_score`, `review_score` 1–5, `hallucinated_objects`, `status` = pending / approved / edited / rejected). This matches the Streamlit app features (approve / reject / edit / filter / export).
- `objects[].caption` and `objects[].instruction` hold the verified object caption and the grounded sorting instruction (Week 3).
- The draft annotation guidelines (caption template, anti-hallucination rules, review-score rubric) are in `docs/annotation_guidelines.md`.

---

## 6. Data pipeline (what was implemented)

| Step | Script | What it does |
|---|---|---|
| 1 | `01_download_taco.py` | Downloads 1,500 TACO images (parallel, resumable, hard timeout), applies EXIF orientation, resizes, rescales boxes and masks, merges scene tags by majority vote |
| 2 | `02_download_zerowaste.py` | Range-request download of 600 evenly spaced frames across all videos, keeping the official splits |
| 3 | `03_download_coco.py` | Extracts only `instances_val2017.json` from the annotation zip, filters and balances images |
| 4 | `04_build_manifest.py` | Cleaning + mapping to the unified schema + location/occlusion/quality flags |
| 5 | `05_make_splits.py` | Stratified train/val/test splits, plus the CSV export |
| 6 | `06_dataset_stats.py` | Statistics tables (`data/reports/stats.md`) + visual sample grid |
| 7 | `07_validate_schema.py` | Validates every record against the JSON Schema, checks files exist and boxes lie inside images |

### Cleaning results (`data/reports/cleaning_report.json`)

| Check | Removed / flagged |
|---|---|
| Download failures (TACO server timeouts) | 5 images removed |
| Below 224 px resolution | 0 |
| Image size ≠ annotation frame (EXIF/rotation errors) | 0 (EXIF fix verified) |
| Degenerate boxes (< 2 px) | 1 annotation removed |
| Boxes slightly outside the image | clipped to the border |
| Exact duplicates (MD5) | 1 image removed (taco_01305) |
| Near-duplicates (perceptual hash, Hamming ≤ 4) | 1 image removed (taco_01157 ≈ taco_01131) |
| Too dark / too bright / blurry | 9 flagged *too_dark* (kept, for review) |
| Tiny objects (< 0.05% of image) | 1,205 flagged `tiny` |
| Possibly occluded (> 30% of box covered by other boxes) | 2,049 flagged `possibly_occluded` |

A visual check (`data/reports/sample_grid.jpg`) confirms that boxes stay aligned after EXIF correction and resizing for all three sources.

### Split strategy
- **ZeroWaste:** official video-level split (prevents near-duplicate frames leaking into test).
- **TACO + COCO:** greedy **multi-label iterative stratification**. Images are assigned rarest-category-first so every category appears in train, val and test in roughly 70/15/15 proportion (hazardous, with only 2 objects, is the exception).

| split | TACO | ZeroWaste | COCO | total |
|---|---:|---:|---:|---:|
| train | 1,063 | 418 | 231 | **1,712** |
| val | 222 | 90 | 43 | **355** |
| test | 208 | 89 | 46 | **343** |

---

## 7. Dataset statistics (pre-captioning)

Full tables are in `data/reports/stats.md`.

**Objects per category:** carton_box 2,627 · unspecified 1,234 · other_litter 1,127 · bag_wrapper 870 · bottle_jar 828 · cup 812 · cigarette 660 · cap_lid 375 · can 273 · straw_utensil 198 · paper 153 · glass_fragment 138 · container_packaging 86 · food_waste 8 · hazardous 2

**Objects per material:** plastic 3,384 · paper_cardboard 2,767 · unknown 1,346 · composite 824 · metal 592 · glass 432 · textile_rubber 36 · organic 8 · e_waste 2

**Clutter:** single 773 · sparse 721 · moderate 742 · dense 174 images

**Environments:** vegetation 754 · sand/dirt 597 · conveyor belt 597 · pavement 433 · unknown 415 (COCO) · indoor 112 · water 84 · clean 37 · trash pile 5

**Object sizes:** small 4,094 · medium 4,371 · large 926

---

## 8. Observations & risks identified

1. **Long tail:** food_waste (8) and hazardous (2) are too rare to evaluate on their own. I plan to merge them into `other_litter` for the metrics, or add a few targeted images.
2. **ZeroWaste labels are incomplete for captioning:** many visible objects (especially paper) are unlabeled background, and object types are missing (`unspecified`). This is a good case for the human-verification step, and a likely source of "hallucination" false alarms that the evaluation must account for.
3. **COCO objects are mostly not waste:** they will be kept as a contrast set and reported separately in evaluation.
4. **Many tiny objects** (especially cigarette butts): object-level captions will be required only for non-tiny objects. Tiny ones still count in scene captions.
5. **Sorting rules are region-dependent:** the defaults follow common municipal rules and are configurable in `taxonomy.yaml`.
6. **License:** ZeroWaste is non-commercial, so the combined dataset will be released under **CC BY-NC 4.0**.
7. Data size on disk: about 0.9 GB (images 810 MB).

---

## 9. Files produced this week

```
README.md                      project overview + how to reproduce
WEEK1_UPDATES.md               this report
requirements.txt
configs/taxonomy.yaml          categories, materials, conditions, actions, all label mappings
schema/annotation_schema.json  JSON Schema v1.0
docs/annotation_guidelines.md  caption style, verification checklist, review rubric (draft v0.1)
scripts/01…07_*.py             full data pipeline
data/processed/                manifest_all.jsonl, train.jsonl, val.jsonl, test.jsonl, objects.csv
data/reports/                  cleaning_report.json, stats.json, stats.md, sample_grid.jpg
data/images/{taco,zerowaste,coco}/   2,410 processed images
```

---

## 10. Plan for Week 2 (Caption Generation)

- Set up BLIP-2 (`Salesforce/blip2-opt-2.7b`) and Qwen2.5-VL (`Qwen/Qwen2.5-VL-7B-Instruct`; 3B if GPU memory is tight), with LLaVA as an optional third model.
- Design prompt templates for image-level captions and **object-level crop/box-prompted captions** that ask for material, condition and location.
- Generate candidates for all 2,410 images, store them in `captions[]` with model and prompt metadata, and compute CLIPScore.
- Start the Streamlit curation app skeleton (image browser, model selector, side-by-side BLIP-2 vs Qwen2.5-VL view).
- **Need:** GPU access (≥ 16 GB VRAM recommended for Qwen2.5-VL-7B).

---

## 11. Update for the professor (copy/paste)

> **Week 1 update – CaptionGene (Waste Image Captioning)**
>
> Week 1 (project setup & dataset preparation) is complete:
>
> 1. **Dataset study:** I analysed TACO (1,500 in-the-wild litter images, 60 classes, masks), ZeroWaste-f (4,503 conveyor-belt frames, 4 material classes, very dense clutter) and COCO (bottle / cup / wine glass). I documented what each one contributes and its limits: TACO is long-tailed, ZeroWaste has material-only labels and near-duplicate video frames, and COCO objects are mostly not waste.
> 2. **Unified taxonomy & schema:** I defined 15 object categories, 9 materials, 13 physical conditions, 8 sorting actions and scene context (environment + clutter level). I explicitly mapped all 60 TACO, 4 ZeroWaste and 3 COCO labels onto it, and formalised the annotation format as a JSON Schema with fields for object category, material, condition, location (box, mask, grid region, size), sorting action, VLM caption candidates and review status.
> 3. **Data collection & cleaning:** I built a reproducible 7-step Python pipeline that downloads, EXIF-corrects, resizes, de-duplicates (MD5 + perceptual hash) and quality-checks the images. The result is **2,410 clean images with 9,391 annotated objects** (TACO 1,493, ZeroWaste 597, COCO 320). All records pass schema validation.
> 4. **Splits:** train / val / test = 1,712 / 355 / 343. ZeroWaste keeps its video-level split to avoid leakage, and TACO/COCO use multi-label stratification so every category appears in all splits.
> 5. **Deliverables:** taxonomy config, JSON schema, draft annotation guidelines, dataset statistics report and a visual sanity-check grid.
>
> **Next week:** generate captions with BLIP-2 and Qwen2.5-VL (object- and image-level), compute CLIPScore, and start the Streamlit curation app.
> **Points for your input:** (a) is the 15-category taxonomy at the right granularity? (b) is it OK to use COCO as a "non-waste" contrast set? (c) could I get GPU access for Qwen2.5-VL-7B?
