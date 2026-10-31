# Week 2 Updates: Caption Generation

**Project:** Vision-Language Caption Generation for Waste Images (CaptionGene)
**Week 2 goal (from plan):** Implement a pretrained vision-language model (VLM) to generate object-level captions describing waste objects, their materials, conditions, and spatial relationships. Generate initial captions for the selected images and associate them with the available bounding boxes or segmentation masks.

**Status: ✅ Pipeline, schema, quality signals and curation app done and tested. ⚠️ The full 2,410-image run with the target models (BLIP-2, Qwen2.5-VL) still needs a GPU.** This week I ran a **60-image pilot** on my laptop with small stand-in models from the same families. All numbers below come from that pilot.

---

## 1. Summary (TL;DR)

| Item | Result |
|---|---|
| Captioning pipeline | 4 new scripts (`08`–`11`) + 2 libraries (`vlm.py`, `captionlib.py`); resumable, config-driven (`configs/captioning.yaml`) |
| Models supported | **BLIP-2**, **Qwen2.5-VL-7B / 3B**, **LLaVA-1.5-7B** (target, GPU); **BLIP-base**, **SmolVLM-256M** (pilot stand-ins, CPU / Apple silicon); **label template** (rule-based baseline) |
| Caption levels | **Image-level** (whole scene) + **object-level** (padded crop of each box, linked by `object_id` to its bbox **and** segmentation mask) |
| Pilot run | 60 val images (30 TACO · 15 ZeroWaste · 15 COCO), **705 candidates**: 180 image-level + 525 object-level (175 objects × 3 models) |
| Quality signals | **CLIPScore** for every candidate + **rule-based auto-checks** (material / category agreement with labels, unsupported objects, people/animals, brand text, conditions) |
| Schema | v1.0 → **v1.1** (object `caption_candidates`, model/prompt provenance, `proposed` conditions, `auto_checks`); all 2,410 records validate with **0 errors** |
| Curation app | `app/streamlit_app.py`: browse/filter, side-by-side models, approve / edit / reject / regenerate, structured fields, generate with any model + custom prompt, statistics, JSONL/CSV export. Tested headlessly with Streamlit `AppTest` |
| Main finding | Small litter is usually **left out** of scene captions, and **CLIPScore does not separate correct from wrong captions** on waste images. Object-level crops + structured checks + human review are needed, not CLIPScore filtering alone |

---

## 2. Compute constraint and how I handled it

My laptop is an **Apple M1 with 8 GB RAM, no CUDA GPU, and only a few GB of free disk**. The planned target models do not fit:

| Model | Download | Memory to run | Fits on M1 8 GB? |
|---|---:|---:|---|
| BLIP-2 OPT-2.7B (fp16) | ~15 GB | ~8 GB VRAM | ❌ |
| Qwen2.5-VL-7B (bf16) | ~16.6 GB | ~17 GB VRAM | ❌ |
| Qwen2.5-VL-3B (bf16) | 7.5 GB | ~8 GB | ❌ (disk + RAM) |
| **BLIP-base** (BLIP family) | 1.0 GB | ~1.5 GB | ✅ |
| **SmolVLM-256M-Instruct** (instruction-tuned, chat template like Qwen) | 0.5 GB | ~1.5 GB | ✅ |
| CLIP ViT-B/32 (for CLIPScore) | 0.6 GB | <1 GB | ✅ |

So I built the pipeline **model-agnostic**: every model is one entry in `configs/captioning.yaml` with two backends, `blip` and `chat`. Then I:

1. Ran the **whole pipeline end-to-end on a 60-image pilot** with the two small stand-ins (one per family), to debug it and get real first results.
2. Wrote `scripts/run_week2_gpu.sh`, which runs the identical pipeline with **BLIP-2 + Qwen2.5-VL** on all 2,410 images on any CUDA GPU (Colab / lab server). Only the model names change.

The stand-ins are much weaker than the target models. The pilot numbers show **how the pipeline and the quality checks behave**. They are **not** the final caption quality of BLIP-2 / Qwen2.5-VL.

---

## 3. Caption generation design

### 3.1 Two caption levels

| Level | Input to the VLM | Linked to | Stored in |
|---|---|---|---|
| Image | whole image | `image_id` | `record.captions[]` |
| Object | crop around the box, enlarged by **15% on each side** for context; tiny crops upscaled to ≥ 32 px | `object_id` → that object's `bbox`, `bbox_norm`, `segmentation`, `location.region` | `record.objects[i].caption_candidates[]` (each keeps the exact `crop_box` used) |

**Which objects get a caption:** objects flagged `tiny` (< 0.05% of the image) are skipped, following guideline rule 7. Of the rest, at most **8 per image (largest first)** are captioned, so dense ZeroWaste frames stay tractable. In the pilot: 205 objects → 22 tiny skipped, 8 skipped by the cap, **175 captioned**.

**Spatial relationships:** each object's 3×3 grid region and size come from its box (Week 1). The object prompt tells the model where the object lies ("in the {region} part of the full photo"), and the template captions state it explicitly ("…in the bottom left of the image"). Pairwise relations ("next to", "on top of") are planned for the Week 3 instruction generation, using the box-overlap data from Week 1.

### 3.2 Prompts (`configs/captioning.yaml → prompts`)

| Prompt id | Used by | Idea |
|---|---|---|
| `blip_image` = "a photo of" | BLIP-base | conditional prefix; the prefix is stripped from the output |
| `blip2_image` = "" | BLIP-2 | BLIP-2 captions best unconditionally |
| `blip2_object` | BLIP-2 | "Question: What is this waste object and what material is it made of? Answer:" |
| `chat_image` | Qwen2.5-VL, LLaVA | asks for items + material + physical state + surface, "only what you can see, no brands" |
| `chat_image_short` | SmolVLM | shorter version (see 3.4: the long one caused prompt leakage) |
| `chat_object_json` | Qwen2.5-VL, LLaVA | gives the dataset label, material and region as a hint and asks for **JSON** `{"caption", "condition": [...]}` restricted to the 13 taxonomy conditions |
| `chat_object` | SmolVLM | free-text version: "Describe the {label} in this photo in a few words, including its colour and condition." |

**Why object prompts include the label:** the boxes already have a verified class, so the VLM only needs to add what the labels lack: **condition, colour and context**. The cost: object-level material/category agreement is no longer an independent test of recognition (see 6.3). Image prompts get **no hints**, so they are a fair test.

### 3.3 Output handling

- **JSON parsing** (`captionlib.parse_object_output`): if the model returns valid JSON, `caption` and the taxonomy-valid `condition` values are used. Otherwise it falls back to free text, and conditions are extracted with a keyword lexicon.
- **Proposed conditions** are stored in `proposed.condition`. They are **not** written into the object's `condition` field. A human accepts them in the app (Week 3).
- **Sentence cleanup:** small chat models ignore "one sentence" and run on until the token limit, leaving a cut-off last sentence. The stored caption keeps at most 2 complete sentences (image) or 1 (object). The full model output is kept in `raw_text`.
- **Decoding:** greedy by default (reproducible). `--num-candidates N` switches to sampling (T = 0.7, top-p 0.9) to get several diverse candidates per image. BLIP uses `no_repeat_ngram_size=3`.

### 3.4 Rule-based template baseline (no VLM)

`template` builds a caption only from the structured labels, e.g. *"Four waste items indoors: two clear plastic bottles, a plastic bottle cap and a single-use carrier bag."* or *"plastic film in the top right of the image"*. It serves three purposes:

1. a **structured-language reference**: what a complete caption should name;
2. a **lower bound** that any useful VLM must beat on condition and context;
3. a **test of the auto-checker**: the template only states label facts, so any "unsupported" flag on it is a checker error. After fixing the lexicon it has **0 false flags on all 60 images**.

### 3.5 Storage and reproducibility

```
data/captions/<model>.jsonl     raw candidates, one row each (append-only, resumable)
data/captions/pilot_ids.txt     the exact 60 pilot images (seed 42)
data/reviews/reviews.jsonl      append-only log of reviewer actions (from the app)
          │
          ▼  scripts/10_merge_captions.py   (rebuilt from scratch each run, so idempotent)
data/processed/manifest_all.jsonl + train/val/test.jsonl + captions.csv
```

Each candidate records `model`, `hf_id`, `prompt_id`, the exact filled `prompt`, generation settings, `runtime_s` and `created_at`. Re-running scripts 04/05 (Week 1) resets the records, and re-running 10 restores all captions and reviews.

---

## 4. Schema v1.1 (`schema/annotation_schema.json`)

New in v1.1, all optional, so v1.0 records still validate:

- `$defs.caption`: one shape shared by image-level `captions[]` and object-level `caption_candidates[]`: `caption_id, text, model, hf_id, prompt_id, prompt, clip_score, status, review_score, hallucinated_objects, reviewer, updated_at, original_text (before a human edit), proposed {condition, parsed_json}, auto_checks {...}`
- `objects[].caption_candidates[]`: object captions, linked to the object's bbox/mask
- `objects[].caption`: now defined as the **latest approved / edited** candidate (filled in Week 3)

`07_validate_schema.py` now also checks that every caption is linked to the right image/object (`caption_id` prefix = `image_id` / `object_id`).
**Result:** `validated 2410 records (705 caption candidates), 0 problems`.

Real object-level candidate from the pilot (abbreviated):

```json
{"caption_id": "taco_01113_o000__smolvlm-256m__chat_object__0",
 "text": "The clear plastic bottle is red with a green label.",
 "model": "smolvlm-256m", "prompt_id": "chat_object",
 "prompt": "Describe the clear plastic bottle in this photo in a few words, including its colour and condition.",
 "clip_score": 0.7703, "status": "pending",
 "proposed": {"condition": [], "parsed_json": false},
 "auto_checks": {"material_match": true, "category_match": true, "mentions_condition": false, "brand_or_text": false}}
```

The caption contradicts itself ("clear … is red"): the model repeats the label hint, then describes the actual colours. Catching this kind of error is the reviewer's job in Week 3.

---

## 5. Quality signals

### 5.1 CLIPScore (`09_clip_score.py`)
Reference-free CLIPScore (Hessel et al., 2021): `2.5 × max(cos(CLIP_image, CLIP_text("A photo depicts " + caption)), 0)` with **CLIP ViT-B/32**. Object captions are scored against the same crop the model saw. BLEU/CIDEr need reference captions, so they come in Week 4, once Week 3 has produced verified captions.

### 5.2 Rule-based auto-checks (`captionlib.image_checks / object_checks`)
Keyword lexicons map caption words to taxonomy values, so each caption can be compared with the labels:

| Check | Meaning |
|---|---|
| `mentioned_materials` / `material_recall` | which labelled materials the caption names |
| `material_conflict` | names a material, but none of the labelled ones (a `composite` label also accepts paper / plastic / metal) |
| `category_recall` | share of labelled object categories the caption names |
| `unsupported_categories` | names a waste category **not in the labels**: hallucination candidate. Turned off when the image has `unspecified` / `other_litter` objects, whose type is unknown |
| `mentions_people_animals` | people/animals are never labelled in these datasets: in TACO/ZeroWaste they are usually hallucinated |
| `brand_or_text` | quotes label text or names a brand (forbidden by the guidelines) |
| `mentions_condition` | uses a condition word (negation-aware: "not fully open" does not count) |

Fixes made while building the lexicon, all found on pilot data: bare "can" excluded ("we **can** see…"), "food can" ≠ food waste, "bottle cap" ≠ bottle, paper cup = `composite`, plurals ("films", "boxes"), negations.

These checks are **signals that tell the reviewer where to look**, not ground truth (see 6.4).

---

## 6. Pilot results (60 val images)

Full tables: `data/reports/captions_week2.md` / `.json`. Example sheet: `data/reports/caption_examples.jpg`.

### 6.1 Image-level captions

| metric | BLIP-base | SmolVLM-256M | template |
|---|---:|---:|---:|
| mean words | 7.0 | 24.6 | 13.1 |
| CLIPScore | **0.747** | 0.745 | 0.703 |
| mentions a material | 23% | 45% | 68% |
| material recall | 0.18 | 0.36 | 0.82 |
| **category recall** | **0.06** | **0.22** | 1.00 |
| unsupported category | 6.7% | 3.3% | 0% |
| people / animals mentioned (TACO + ZeroWaste) | **13.3%** | 2.2% | 0% |
| mentions a condition | 6.7% | 6.7% | 0% |
| brand / label text | 0% | 1.7% | 0% |
| CLIPScore: COCO / TACO / ZeroWaste | 0.76 / 0.75 / 0.74 | 0.74 / 0.74 / 0.76 | 0.55 / 0.75 / 0.79 |

### 6.2 Object-level captions (175 objects)

| metric | BLIP-base | SmolVLM-256M | template |
|---|---:|---:|---:|
| mean words | 6.5 | 14.9 | 9.4 |
| CLIPScore | 0.655 | **0.674** | 0.611 |
| material match | 20% | 96%\* | 100% |
| category match | 13% | 99%\* | 100% |
| mentions a condition | 0.6% | 17.7% | 0% |
| brand / label text | 0% | 3.4% | 0% |
| sec / caption (M1, generation only) | 0.29 | 0.58 | – |

\* inflated: the object prompt contains the label (3.2).
**Proposed conditions (SmolVLM):** intact 23, torn 5, soiled 2, crumpled 1.
**CLIPScore head-to-head** (same image/object): image level BLIP 29 vs SmolVLM 31; object level 75 vs 100.

### 6.3 Findings

1. **Small waste is left out of scene captions.** Most TACO images show a few small items on sand or grass, and both models describe the background: *"the sand at the beach"*, *"a leaf litter on the ground"*. Category recall at image level is only 0.06 (BLIP) and 0.22 (SmolVLM). This is the main reason for **object-level captioning on crops**.
2. **CLIPScore is not a good quality filter for waste captions.** The template, correct by construction, scores *lower* than BLIP on COCO and TACO. BLIP and SmolVLM have almost the same CLIPScore (0.747 vs 0.745) although SmolVLM names 3× more of the labelled objects. CLIP rewards describing the dominant scene ("a plate of food with a fork and knife") over the small waste object. Week 3 will therefore use CLIPScore only for ranking/sorting, not as a hard threshold.
3. **Hallucinations are often non-waste objects.** An informal qualitative spot check of 12 pilot images (24 VLM captions, looked at during development; this is not the Week 3 human verification) found *"the body of a man found in the jungle"* (a plastic-bag scene), *"a dead animal lying on the sand"*, *"a seat with a stuffed animal on it"* and *"a beach with a surf board"*. The waste-category check caught only 1 of these images, which is why I added the people/animals flag. BLIP mentions people or animals in **13%** of TACO/ZeroWaste captions.
4. **Instruction-following models can be steered toward the structured format, but small ones fall into stock phrases.** SmolVLM gives material + colour + condition when asked, but on cardboard crops it repeated the stock sentence *"The box is open, revealing a glimpse of…"* 20+ times. Its long first prompt (with example words "crushed, torn, dirty") also made it repeat those words for every image (**prompt leakage**), so I shortened the prompt.
5. **Brand / label text** shows up (*"has the word 'Lucky' on the front"*, *"Fairy Original hand soap"*) and is now flagged automatically.
6. **ZeroWaste:** "a pile of paper and other items" is *correct*. The frames are full of unlabeled newspaper (Week 1, risk 2). Checks are relaxed on images with `unspecified` objects so this is not marked as a hallucination.
7. **COCO works as a contrast set:** models describe meals and tables, not waste. The template was fixed to say "item", not "waste item", when `is_discarded` is not "yes".

### 6.4 What the numbers can and cannot say
- Auto-checks are **keyword rules**. They have **0 false flags on the 60 label-derived template captions**, but they **miss** hallucinations outside their word lists (finding 3). Measured precision/recall against human labels comes in Week 3.
- Only 60 images and two small stand-in models: differences of a few points are noise.

---

## 7. Streamlit curation app (`streamlit run app/streamlit_app.py`)

| Required feature (project brief) | Status |
|---|---|
| Upload/select waste images and display metadata | ✅ select + filter (split, source, category, review status, pending, auto-check flags, min CLIPScore), prev/next; metadata panel with boxes / masks overlay. *File upload of new images: planned for Week 4* |
| Select a captioning model and enter a prompt template | ✅ Generate tab: any model in the config, editable prompt template with placeholders, preview of the filled prompt |
| Generate one or multiple candidate captions | ✅ 1–5 candidates (sampling when > 1), scored with CLIPScore and stored immediately |
| Edit, approve, reject, or regenerate captions | ✅ per candidate, plus "write your own caption" |
| Structured fields: object, material, condition, count, scene context | ✅ Objects tab (category, material, condition multi-select with "accept VLM conditions", sorting action, per-category count) + scene form (environment, discarded?, scene description, image status, notes) |
| Quality score / review score and filtering | ✅ CLIPScore + auto-check badges, review score 1–5, hallucinated-object tags, filters |
| Side-by-side BLIP-2 and Qwen2.5-VL | ✅ configurable compare pair (`compare_pair: [blip2, qwen2.5-vl-7b]`; falls back to the pilot pair until the GPU run exists) |
| Export approved annotations as JSON/JSONL and CSV | ✅ JSONL (image + object) and two CSVs, filtered by status and split |
| Statistics: images, class distribution, approval rate, captions per class | ✅ Statistics tab |

**Design:** the app never edits the dataset files. Every action is appended to `data/reviews/reviews.jsonl` with reviewer name + timestamp, so reviews are auditable and later usable for inter-annotator agreement. Script 10 merges them into the dataset.
**Testing:** run headlessly with `streamlit.testing.v1.AppTest`: page load, navigation, approve, structured-field save, export buttons and live generation with SmolVLM + CLIP all ran without exceptions. Test review events and the test generation were deleted afterwards, so **no review has been recorded yet**.

---

## 8. Engineering issues solved this week

| Problem | Fix |
|---|---|
| `pip` installed transformers **5.21** (new API: `dtype=` not `torch_dtype=`, no `AutoModelForVision2Seq`, CLIP feature functions return an output object) | wrappers use `AutoModelForImageTextToText` + `dtype`; `ClipScorer` reads `pooler_output`. `requirements.txt` documents the tested versions |
| protobuf 7 broke Streamlit 1.45 | pinned `protobuf<7` |
| BLIP repo has no safetensors on `main` → both `.bin` and safetensors were downloaded (≈1 GB extra on a nearly full disk) | deleted the duplicate; load CLIP with `use_safetensors=True` |
| 8 GB RAM, heavy swapping (generation ~5× slower than in isolation) | small models, batched generation, resumable output files |
| BLIP decodes "close-up" as "close - up", so the prompt prefix was not stripped | hyphen normalisation before stripping |
| BLIP loops ("a bag containing a bag containing…") | `no_repeat_ngram_size=3` |
| SmolVLM: prompt leakage, ignoring JSON format, run-on text | shorter prompt, free-text object prompt + parser, sentence cleanup with `raw_text` kept |

---

## 9. How to run

```bash
pip install -r requirements.txt
cd scripts

# local pilot (CPU / Apple silicon, ~20 min on an 8 GB M1)
python 08_generate_captions.py --models template blip-base smolvlm-256m --subset pilot
python 09_clip_score.py
python 10_merge_captions.py
python 11_caption_report.py
python 07_validate_schema.py

# full run with the target models on a CUDA GPU
bash run_week2_gpu.sh                      # all 2,410 images, BLIP-2 + Qwen2.5-VL-7B
QWEN=qwen2.5-vl-3b bash run_week2_gpu.sh   # 16 GB GPU (e.g. Colab T4)

# Google Colab route (no local GPU): resumable, captions saved to Google Drive after every batch
bash make_colab_bundle.sh                  # zip for upload to MyDrive/CaptionGene/
#   notebooks/week2_colab.ipynb in Colab -> captions_out.zip
bash import_colab_results.sh ~/Downloads/captions_out.zip   # pilot captions archived to data/captions/pilot/

# app
cd .. && streamlit run app/streamlit_app.py
```

Other options: `--subset split:test`, `--subset ids:<file>`, `--level image|object`, `--num-candidates 3`, `--limit N`.

---

## 10. Files added / changed this week

```
WEEK2_UPDATES.md                 this report
configs/captioning.yaml          NEW  model registry, prompts, generation, crop + pilot settings
schema/annotation_schema.json    v1.0 -> v1.1
docs/annotation_guidelines.md    v0.1 -> v0.2 (section 5: reviewing VLM candidates in the app)
scripts/vlm.py                   NEW  BLIP / chat-VLM wrappers + ClipScorer
scripts/captionlib.py            NEW  crops, prompts, parsing, cleanup, template captions, auto-checks, merge
scripts/08_generate_captions.py  NEW
scripts/09_clip_score.py         NEW
scripts/10_merge_captions.py     NEW
scripts/11_caption_report.py     NEW
scripts/run_week2_gpu.sh         NEW
scripts/make_colab_bundle.sh     NEW  zip for the Colab run
scripts/import_colab_results.sh  NEW  import Colab captions, re-merge, report
notebooks/week2_colab.ipynb      NEW  Colab notebook (T4 GPU)
scripts/07_validate_schema.py    + caption-link checks
app/streamlit_app.py             NEW  curation app
requirements.txt, README.md      updated
data/captions/                   blip-base / smolvlm-256m / template .jsonl (705 candidates), pilot_ids.txt
data/processed/                  manifest + splits now carry captions; NEW captions.csv
data/reports/                    NEW captions_week2.md, captions_week2.json, caption_examples.jpg
```

---

## 11. Limitations and risks

1. **Target models not run yet** (GPU needed). The pilot shows the pipeline works and what the failure modes look like, not the final BLIP-2 / Qwen2.5-VL quality.
2. **Object-level agreement is inflated** by the label hint in the prompt. To measure recognition, run an object prompt without the hint on the test split (`--subset split:test`). That is planned for the Week 4 baseline evaluation.
3. **Auto-checks are lexical** and miss paraphrases and non-waste hallucinations. Their precision/recall will be measured against Week 3 human labels.
4. **Conditions are thin:** SmolVLM proposed a condition for only 18% of objects. Qwen's JSON prompt with a fixed condition list should do better; to be confirmed on the GPU run.
5. **Single annotator:** only one reviewer so far. A second reviewer on a ~10% overlap would allow measuring agreement (Cohen's κ).

---

## 12. Plan for Week 3 (Annotation verification and instruction generation)

- Run `run_week2_gpu.sh` (BLIP-2 + Qwen2.5-VL) on all images, or at least val + test.
- Human verification in the app: approve / edit / reject, hallucination tags, accept or fix the proposed conditions. Priority order: flagged captions first, then sorted by CLIPScore.
- Apply the condition-based sorting overrides (`taxonomy.yaml → sorting_overrides`, e.g. soiled cardboard → landfill).
- Generate **grounded sorting instructions** per object ("Pick up the crushed plastic bottle at the bottom-left and place it in the plastic recycling bin"), each linked to one `object_id` (box + mask).
- Measure auto-check precision/recall and hallucination rate against the human labels.

---

## 13. Update for the professor (copy/paste)

> **Week 2 update – CaptionGene (Waste Image Captioning)**
>
> Week 2 (caption generation) is done at the pipeline level and tested on a pilot:
>
> 1. **Captioning pipeline:** a config-driven, resumable pipeline that generates **image-level captions** and **object-level captions** (from a padded crop of each annotated box, linked by `object_id` to its bounding box and segmentation mask). It supports BLIP-2, Qwen2.5-VL (7B/3B) and LLaVA, plus a rule-based "label template" baseline. Instruction-tuned models are asked for structured output (caption + taxonomy conditions as JSON).
> 2. **Quality signals:** CLIPScore for every caption, plus automatic checks against the labels: material and category agreement, unsupported objects (hallucination candidates), people/animals, brand text, and proposed object conditions.
> 3. **Pilot:** my laptop (M1, 8 GB, no GPU) cannot run BLIP-2 or Qwen2.5-VL, so I ran the full pipeline on 60 validation images with small models from the same families (BLIP-base, SmolVLM-256M): 705 candidates, all passing schema validation. Main findings: (a) small litter is usually left out of scene captions (only 6–22% of labelled object types are named), which supports object-level captioning; (b) **CLIPScore does not separate good from bad waste captions**: a caption built from the ground-truth labels scores lower than BLIP's generic scene caption; (c) hallucinations are often non-waste objects ("the body of a man", "a dead animal"), and small models fall into stock phrases.
> 4. **Curation app:** a Streamlit app with side-by-side model comparison, approve/edit/reject/regenerate, structured fields (category, material, condition, count, scene context), generation with any model and custom prompts, statistics, and JSONL/CSV export. All reviewer actions are logged for auditability.
>
> **Next week:** run BLIP-2 + Qwen2.5-VL on the full set, human verification in the app, and grounded sorting instructions per object.
> **Points for your input:** (a) could I get GPU access (≥ 16 GB VRAM; Qwen2.5-VL-7B needs ~17 GB)? (b) is it fine to give the object label as a hint in object prompts (more accurate captions), with an un-hinted run on the test split for evaluation? (c) given finding (b), should CLIPScore be used only for ranking and not as a filter threshold?
