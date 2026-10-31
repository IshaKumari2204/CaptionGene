# CaptionGene: The Whole Project Explained From Zero

> **Who this is for:** someone who has never seen this project and has to explain it to a professor today.
> Read it top to bottom once (about 45 minutes). Then read **Part 9 (likely questions)** and **Part 10 (a 5-minute speech)** again right before the meeting.
>
> Every number in this document comes from the real files in this repository.

---

## Table of contents

- **Part 0:** The project in 60 seconds
- **Part 1:** The problem we are solving
- **Part 2:** Basic concepts you must know (from zero)
- **Part 3:** The plan (5 weeks) and where we are now
- **Part 4:** Week 1, building the dataset (done)
- **Part 5:** Week 2, generating captions with AI models (done)
- **Part 6:** The Streamlit curation app
- **Part 7:** Results and what we learned
- **Part 8:** Limitations, and what comes next
- **Part 9:** Questions your professor may ask, with answers
- **Part 10:** A 5-minute speech you can give
- **Part 11:** How to demo it live
- **Part 12:** A map of every file
- **Part 13:** Glossary

---

# Part 0: The project in 60 seconds

**One sentence:** We are building a **high-quality dataset of waste images with text descriptions** ("captions"). AI models write the first drafts of the descriptions, and a human checks and fixes them.

A bit longer:

1. There are public datasets with **photos of trash** (bottles, cans, wrappers…). They say *where* each object is (a box drawn around it) and *what class* it is ("Clear plastic bottle"). They have **no sentences** describing the scene.
2. AI models that can "look" at images and write text (**vision-language models**, VLMs) are good at general photos but bad at trash. They skip small items, guess the wrong material, invent objects that are not there, and rarely say whether something is **crushed, torn, or dirty**.
3. So we:
   - **Week 1:** collected and cleaned 2,410 waste images from 3 public datasets, and designed a standard way to label each object (category, material, condition, location, which bin it goes in).
   - **Week 2:** wrote code that runs AI models on these images to **automatically write captions**, both for the whole image and for each single object. The code then **scores** each caption automatically and flags suspicious ones. We also built a **web app** where a human can review, edit, approve, or reject the captions.
   - **Weeks 3–5 (future):** humans verify the captions, write sorting instructions ("put this bottle in the plastic bin"), evaluate models, and publish the dataset.
4. **Why it matters:** recycling robots and smart sorting systems need AI that understands trash. To train or test such AI, you need exactly this kind of dataset, and it barely exists today.

---

# Part 1: The problem we are solving

## 1.1 The real-world motivation

Recycling facilities (called **MRFs**, material recovery facilities) receive mixed trash on conveyor belts. Sorting it is slow, dirty, and often done by hand. Robots and cameras can help, but the AI behind them must understand things like:

- **What** the object is (bottle, can, carton…)
- **What it is made of** (plastic, metal, glass, paper…). This decides the bin.
- **What condition** it is in (crushed, torn, dirty…). This also matters: a **greasy pizza box** cannot be recycled as paper.
- **Where** it is in the image, so a robot arm can grab it.

## 1.2 Why existing AI is not enough

Modern AI models such as **BLIP**, **LLaVA**, and **Qwen2.5-VL** can look at an image and write a sentence about it. For a normal photo, e.g. "a dog playing in a park", they do well. For waste images they struggle because:

- Waste images are **cluttered**: many objects overlapping each other.
- Objects are **damaged or deformed**: a crushed can doesn't look like a can.
- Objects are **partially hidden** (occluded).
- Objects are often **tiny**: a cigarette butt on a beach.
- The models were trained on general internet photos, not trash.

So their captions are **incomplete** (they miss objects), **inaccurate** (wrong material), or **hallucinated** (they mention things that are not in the image).

## 1.3 Why we can't just use an existing dataset

Public waste datasets (TACO, ZeroWaste) have **images + boxes + class labels**, but **no captions**. There is very little data that pairs waste images with good, verified text descriptions.

## 1.4 Our goal (from the problem statement)

> Build a **verified waste image–caption dataset**. Generate captions with pretrained VLMs, then refine them with structured quality checks and human review. The captions should describe **waste category, material, physical state (condition), and scene context**.

The four official objectives:

| Objective | In simple words | Status |
|---|---|---|
| **Data aggregation** | Collect waste images from TACO, ZeroWaste, COCO | ✅ Week 1 |
| **Caption generation** | Use AI models to write captions automatically | ✅ Week 2 (pipeline + pilot run) |
| **Quality control** | Measure caption quality (CLIPScore, automatic checks, human review) | ✅ automatic part done; human review is Week 3 |
| **Structured language** | Captions follow a pattern like "crushed plastic bottle" (condition + material + object) | ✅ taxonomy + templates + structured prompts |

**Final deliverable:** a **Streamlit app** to generate, review, edit, filter, and export captions. It is built (see Part 6).

---

# Part 2: Basic concepts you must know (from zero)

Read this part slowly. Everything later builds on it.

### 2.1 What is "image captioning"?
Giving a computer an image and getting back a sentence describing it.
Image → 🤖 → *"A crushed plastic bottle lying on the grass."*

### 2.2 What is a Vision-Language Model (VLM)?
An AI model that takes **an image (vision)** and produces or understands **text (language)**. It has two parts:
- an **image encoder**, which turns the picture into numbers the model understands;
- a **language model**, which writes text based on those numbers.

The models in this project:

| Model | Made by | Type | What it does well |
|---|---|---|---|
| **BLIP** / **BLIP-2** | Salesforce | "captioning" model | Short, generic captions ("a bottle on the ground"). Can't follow detailed instructions well. |
| **Qwen2.5-VL** | Alibaba | "instruction-following" (chat) model | You can *ask it questions* or give instructions ("describe the material…"). Much smarter, but big. |
| **LLaVA** | Research community | chat model | Similar idea to Qwen, older. |
| **SmolVLM-256M** | Hugging Face | tiny chat model | Small enough for a laptop. Our **stand-in** for Qwen. |
| **BLIP-base** | Salesforce | small BLIP | Our **stand-in** for BLIP-2. |
| **CLIP** | OpenAI | matching model (not a captioner) | Measures how well a text matches an image. Used for **CLIPScore**. |

"7B", "2.7B", "256M" = the number of **parameters** (internal numbers the model learned): 7 billion, 2.7 billion, 256 million. More parameters usually means smarter, but it needs more memory and a GPU.

### 2.3 What is a "prompt"?
The text instruction you give a chat model, e.g. *"Describe this image in one sentence. Mention each piece of rubbish and what it is made of."*
Different prompts → different captions. Designing prompts is called **prompt engineering**.

### 2.4 What is "hallucination"?
When an AI **describes something that is not in the image**. Example from our own results: an image of plastic bags in bushes was captioned *"the body of a man found in the jungle"*. There is no man. That's a hallucination. Reducing hallucinations is a key goal.

### 2.5 What is a bounding box?
A rectangle drawn around an object, stored as 4 numbers: **[x, y, width, height]** in pixels.
`[394.7, 616.5, 312.3, 233.6]` = starts 394.7 px from the left and 616.5 px from the top, and is 312.3 px wide and 233.6 px tall. This format is called **COCO format**.

### 2.6 What is a segmentation mask?
A more precise outline of the object: a **polygon** (a list of points) that follows the object's exact shape instead of a rectangle. Our datasets provide both boxes and masks.

### 2.7 What is "object-level" vs "image-level"?
- **Image-level caption:** describes the whole picture. *"Three pieces of litter on a sidewalk: a can, a bottle, and a cigarette."*
- **Object-level caption:** describes **one specific object** (one box). *"Crushed aluminium drink can."* It is linked to that object's box, so a robot knows *which* object the text is about. This linking is called **grounding**.

### 2.8 What is a dataset "split"? (train / validation / test)
We divide the data into three groups:
- **Train (~70%):** used to teach (train) a model.
- **Validation (~15%):** used to tune settings while developing.
- **Test (~15%):** kept hidden until the end, for a fair final score.
If test images leak into training, the scores look better than they really are. That's called **data leakage**.

### 2.9 What is a taxonomy?
A fixed, agreed list of categories. For example: "the only allowed materials are plastic, metal, glass, paper_cardboard, organic, composite, textile_rubber, e_waste, unknown". This keeps everyone's labels consistent. Ours is in `configs/taxonomy.yaml`.

### 2.10 What is a schema? What is JSON / JSONL / CSV?
- **JSON:** a text format for structured data: `{"material": "plastic", "condition": ["crushed"]}`.
- **JSONL ("JSON Lines"):** one JSON object per line, i.e. one image per line. Easy to process big datasets line by line.
- **CSV:** a spreadsheet-like table (opens in Excel).
- **Schema** (JSON Schema): a formal set of rules describing what a valid record looks like (which fields must exist, which values are allowed). A script **validates** every record against it to catch mistakes.

### 2.11 How do we measure caption quality? (metrics)
- **CLIPScore:** uses the CLIP model to measure how well a caption *matches* an image, with **no reference caption needed**. Range 0 to 2.5; good captions are usually around 0.7–0.9. Formula: `2.5 × max(cosine_similarity(image, "A photo depicts " + caption), 0)`. ("Cosine similarity" = how similar two lists of numbers are, from -1 to 1.)
- **BLEU / CIDEr:** compare a generated caption to a **human-written reference caption** by counting overlapping words. They need verified reference captions, so we use them in Week 4.
- **Human review:** a person scores the caption 1–5. The most reliable method, but slow.

### 2.12 What is a GPU and why does it matter here?
A **GPU** (graphics card) runs AI models much faster than a normal processor, and has its own memory (**VRAM**). Large models like Qwen2.5-VL-7B need ~17 GB of VRAM. Our laptop (Apple M1, 8 GB RAM total) cannot run them. This is why Week 2 used small "stand-in" models (Part 5.2).

### 2.13 What is Hugging Face / transformers / PyTorch?
- **PyTorch:** the Python library that runs neural networks.
- **Hugging Face:** a website hosting thousands of pretrained models (like an app store for AI models).
- **transformers:** Hugging Face's Python library to download and run those models in a few lines of code.

### 2.14 What is Streamlit?
A Python library for building simple **web apps** without writing HTML/JavaScript. Our curation app is built with it.

---

# Part 3: The plan (5 weeks) and where we are now

| Week | Task | Status |
|---|---|---|
| **1** | Study datasets, decide categories & annotation format, collect/clean images, make splits, define schema | ✅ **Done** |
| **2** | Use VLMs to generate captions (image + object level), link them to boxes/masks | ✅ **Done** (pipeline + 60-image pilot; full run needs a GPU) |
| 3 | Humans verify/correct captions, remove hallucinations; write grounded sorting instructions | ⏳ Next |
| 4 | Package the final JSON dataset; evaluate a baseline VLM (caption quality, hallucination rate, grounding accuracy) | ⏳ |
| 5 | Error analysis, final splits, dataset card, guidelines, GitHub, report, demo video | ⏳ |

**The full flow of the project, as a picture:**

```
 Public datasets           Week 1                     Week 2                       Week 3+
 ───────────────     ─────────────────────     ───────────────────────     ─────────────────────
 TACO  (litter)  ─┐  download, clean,           AI models write             humans check & fix
 ZeroWaste (MRF) ─┼─► dedupe, unify labels ──►  captions for images  ──►    in the Streamlit app
 COCO (bottles)  ─┘  into one schema,           and for each object,        → verified dataset
                     split train/val/test       auto-score + flag           → sorting instructions
                                                suspicious ones             → evaluation, release
```

---

# Part 4: Week 1, building the dataset (done)

## 4.1 The three source datasets

### TACO ("Trash Annotations in Context")
- **1,500 phone photos of litter in the wild**: beaches, parks, streets, indoors.
- Each object has a **box + segmentation mask** and one of **60 fine-grained classes** (e.g. "Clear plastic bottle", "Drink can", "Crisp packet", "Cigarette").
- Also has **scene tags** (Vegetation, Pavement, Sand, Water…).
- Good: detailed classes, real messy scenes. Bad: small, many rare classes (only 2 batteries!), many tiny objects.
- License: CC BY 4.0 (free to use, give credit).

### ZeroWaste (ZeroWaste-f, a CVPR 2022 paper)
- **Video frames from a real recycling-facility conveyor belt.** Very dense, crushed, overlapping trash.
- Only **4 classes, and they are materials not objects**: cardboard, soft plastic, rigid plastic, metal.
- Many objects (especially paper) are **not labelled at all**.
- Consecutive video frames look almost identical (near-duplicates).
- License: CC BY-NC 4.0 (**non-commercial only**).

### COCO (a famous general dataset)
- COCO has no "trash" or "can" classes. The waste-like classes are **bottle, cup, wine glass**.
- In COCO these objects are mostly **in use** (on a dinner table), *not* trash.
- **Role in our project:** a **contrast set / hard negatives**. It tests whether a model wrongly calls every bottle "waste".

## 4.2 Scope decisions

- **Total: 2,410 clean images.** TACO 1,493 + ZeroWaste 597 + COCO 320.
- **9,391 labelled objects** (average 3.9 per image, max 90 in one image).
- ZeroWaste: only ~600 frames, **evenly spaced in time** to avoid near-duplicates. Downloaded using **HTTP range requests**, i.e. only the needed pieces of a 7.5 GB zip file, never the whole thing.
- COCO: images where a bottle/cup/wine glass covers at least 1% of the image.
- All images resized so the longest side is ≤ 1280 px, rotated correctly using EXIF data (phone orientation info), and boxes/masks rescaled to match.
- "In-the-wild images from internet search": **postponed**, because of license risk and no boxes. The schema still supports `source: "wild"` for later.

## 4.3 The unified taxonomy (`configs/taxonomy.yaml`)

The three datasets use different label systems, so we created **one common system**. Every object gets **four independent labels**:

| Field | Question it answers | Allowed values |
|---|---|---|
| **category** (15) | *What is it?* | bottle_jar, can, carton_box, cup, bag_wrapper, container_packaging, cap_lid, paper, straw_utensil, cigarette, food_waste, glass_fragment, hazardous, other_litter, unspecified |
| **material** (9) | *What is it made of?* | plastic, metal, glass, paper_cardboard, organic, composite, textile_rubber, e_waste, unknown |
| **condition** (13, can have several) | *What state is it in?* | intact, crushed, dented, crumpled, torn, broken, open, soiled, wet, burnt, faded, partially_occluded, heavily_occluded |
| **sorting_action** (8) | *Which bin?* | recycle_plastic, recycle_metal, recycle_paper, recycle_glass, compost, landfill, hazardous_dropoff, needs_review |

Why separate fields? So captions can be **composed in a structured way**: `<condition> <material> <category>` → *"crushed plastic bottle"*. This is the "structured language" objective.

**Mapping:** every original label was mapped by hand. All 60 TACO, 4 ZeroWaste, and 3 COCO classes. Examples:

| Original label | category | material | default bin |
|---|---|---|---|
| TACO "Clear plastic bottle" | bottle_jar | plastic (PET) | recycle_plastic |
| TACO "Drink carton" | carton_box | composite (Tetra Pak) | recycle_paper |
| TACO "Crisp packet" | bag_wrapper | composite (metallised) | landfill |
| ZeroWaste "soft_plastic" | **unspecified** (object type unknown) | plastic (film) | landfill |
| COCO "bottle" | bottle_jar | **unknown** (could be glass or plastic) | needs_review |

**Special rules (overrides):** e.g. *soiled paper/cardboard → landfill* (greasy pizza box), *broken glass → landfill*.

**Other info stored per object/image:**
- **Location:** the image is divided into a 3×3 grid (top-left, center, bottom-right…), plus a **size**: small (<1% of image), medium (<10%), large (≥10%).
- **Scene:** environment (vegetation, sand, pavement, water, conveyor_belt, indoor…), **clutter level** (single = 1 object, sparse = 2–3, moderate = 4–9, dense = 10+), and `is_discarded` (yes/no/unknown; COCO = unknown).
- **Flags:** `tiny` (smaller than 0.05% of the image) and `possibly_occluded` (more than 30% of its box is covered by other boxes).

## 4.4 The annotation schema (`schema/annotation_schema.json`)

**One record per image** (one line in a JSONL file). Simplified:

```json
{
  "image_id": "taco_00001",  "source": "taco",  "split": "val",
  "file_name": "images/taco/taco_00001.jpg",  "width": 960, "height": 1280,
  "scene": { "environment": ["indoor_manmade"], "clutter_level": "sparse",
             "object_count": 2, "is_discarded": "yes", "description": null },
  "objects": [ {
      "object_id": "taco_00001_o000",
      "source_category": "Meal carton",
      "category": "carton_box", "material": "paper_cardboard",
      "condition": [], "sorting_action": "recycle_paper",
      "bbox": [0.6, 285.4, 892.5, 948.8],
      "segmentation": [[ ...polygon points... ]],
      "location": {"region": "center", "size": "large"},
      "flags": [],
      "caption": null,               <- verified object caption (Week 3)
      "caption_candidates": [...],   <- AI-generated object captions (Week 2)
      "instruction": null            <- sorting instruction (Week 3)
  } ],
  "captions": [...],                 <- AI-generated image captions (Week 2)
  "review": {"status": "unreviewed", "reviewer": null}
}
```

## 4.5 The Week 1 data pipeline (7 scripts in `scripts/`)

| # | Script | What it does |
|---|---|---|
| 1 | `01_download_taco.py` | Downloads 1,500 TACO images, fixes rotation, resizes, rescales boxes/masks |
| 2 | `02_download_zerowaste.py` | Downloads only ~600 selected frames from the 7.5 GB zip (range requests) |
| 3 | `03_download_coco.py` | Pulls only the needed COCO annotation file; selects bottle/cup/glass images |
| 4 | `04_build_manifest.py` | **Cleans** everything and converts to our schema (details below) |
| 5 | `05_make_splits.py` | Makes train/val/test splits + a CSV with one row per object |
| 6 | `06_dataset_stats.py` | Statistics tables + a visual sample grid |
| 7 | `07_validate_schema.py` | Checks every record against the schema, every file exists, every box is inside its image |

**Cleaning steps (script 04):**
1. Remove images that failed to download (5 TACO server timeouts).
2. Remove images smaller than 224 px (0 found).
3. Remove images whose size doesn't match their annotations, which would mean misaligned boxes (0 after the rotation fix).
4. Remove broken boxes smaller than 2 px (1 removed). Clip boxes slightly outside the image.
5. Remove **exact duplicates** (MD5 file hash; 1 removed) and **near-duplicates** (perceptual hash, an "image fingerprint" that is similar for similar-looking images; 1 removed).
6. **Flag but keep** blurry/dark/bright images (9 flagged too dark), tiny objects (1,205), possibly occluded objects (2,049).

**Splits (script 05):**
- ZeroWaste keeps its **official split, which is done by video**. Frames of the same video are near-identical, so they must never be split between train and test (that would be leakage).
- TACO + COCO use **multi-label stratification**: every category appears in train, val, and test in roughly 70/15/15 proportion. Images with the rarest categories are assigned first.
- Result: **train 1,712 / val 355 / test 343.**

**Week 1 statistics (highlights):**
- Most common categories: carton_box 2,627 · unspecified 1,234 · other_litter 1,127 · bag_wrapper 870 · bottle_jar 828 · cup 812 · cigarette 660.
- Very rare: food_waste 8 · hazardous 2 (the "long tail" problem).
- Materials: plastic 3,384 · paper_cardboard 2,767 · unknown 1,346 · composite 824 · metal 592 · glass 432.
- Validation: **0 errors on all 2,410 records.**

**Annotation guidelines** (`docs/annotation_guidelines.md`): rules for humans, e.g. "only describe what is visible", "no brand names", "if unsure of the material, say 'bottle' not 'glass bottle'", the caption template, and the 1–5 review score rubric.

---

# Part 5: Week 2, generating captions with AI models (done)

Week 2 goal (from the plan): *"Implement a pretrained VLM to generate object-level captions describing waste objects, their materials, conditions, and spatial relationships. Generate initial captions for the selected images and associate them with available bounding boxes or segmentation masks."*

## 5.1 The big picture of Week 2

```
 configs/captioning.yaml  (which models, which prompts, which settings)
          │
          ▼
 08_generate_captions.py ── runs AI models ──► data/captions/<model>.jsonl  (raw captions)
          │                                          │
          ▼                                          ▼
 09_clip_score.py ── adds a CLIPScore to every caption
          │
          ▼
 10_merge_captions.py ── puts captions + human reviews into the dataset records,
          │               runs automatic checks on each caption
          ▼
 11_caption_report.py ── statistics report + example picture
          │
          ▼
 07_validate_schema.py ── checks everything is still valid

 app/streamlit_app.py ── the web app where humans review everything
```

## 5.2 The hardware problem, and the honest solution

The plan was to use **BLIP-2** and **Qwen2.5-VL-7B**. Our laptop is an **Apple M1 with 8 GB RAM and no NVIDIA GPU**, and its disk was nearly full.

| Model | Download size | Memory needed | Runs on our laptop? |
|---|---|---|---|
| BLIP-2 (OPT-2.7B) | ~15 GB | ~8 GB GPU | ❌ |
| Qwen2.5-VL-7B | ~16.6 GB | ~17 GB GPU | ❌ |
| Qwen2.5-VL-3B | 7.5 GB | ~8 GB | ❌ |
| **BLIP-base** | 1 GB | small | ✅ (stand-in for BLIP-2) |
| **SmolVLM-256M** | 0.5 GB | small | ✅ (stand-in for Qwen2.5-VL) |
| CLIP ViT-B/32 | 0.6 GB | small | ✅ (for CLIPScore) |

**What we did:**
1. Wrote the code so that **switching models only means changing a name** in a config file.
2. Ran the **entire pipeline on a 60-image "pilot"** with the two small models (one from each family: a BLIP-type and a chat-type). This proves the pipeline works and gives first real results.
3. Wrote `scripts/run_week2_gpu.sh`, which runs **the exact same pipeline with BLIP-2 + Qwen2.5-VL on all 2,410 images** on any GPU computer (Google Colab, lab server).

**Be honest when presenting:** the pilot numbers show how the pipeline and checks behave. They are *not* the final quality of BLIP-2/Qwen, which are much stronger models.

**The pilot set:** 60 images from the **validation split**: 30 TACO, 15 ZeroWaste, 15 COCO, randomly chosen with a fixed seed (42), so the result is reproducible. The list is saved in `data/captions/pilot_ids.txt`.

## 5.3 The configuration file (`configs/captioning.yaml`)

All settings live in one file instead of being hard-coded:
- **models:** each model has a name, its Hugging Face ID, its type ("backend": `blip` or `chat`), precision, and which prompts to use.
- **prompts:** the prompt texts.
- **generation:** how many words max (60 tokens), greedy decoding (deterministic, so you get the same output every run), optional multiple candidates.
- **object_crop:** 15% padding around boxes, skip tiny objects, max 8 objects per image.
- **pilot:** how many images per source.
- **compare_pair:** which two models the app shows side by side (BLIP-2 vs Qwen2.5-VL).

## 5.4 The model wrapper (`scripts/vlm.py`)

Different models are called differently. This file hides that, so the rest of the code just says `model.generate(images, prompts)`.
- **BlipCaptioner:** for BLIP/BLIP-2. You give an image plus an optional text **prefix** like "a photo of", and the model continues the sentence. We then remove the prefix.
- **ChatCaptioner:** for Qwen, LLaVA, SmolVLM. Builds a chat message (image + instruction) in each model's own format (its "chat template").
- **ClipScorer:** computes CLIPScore.
- Automatically uses the best hardware available: NVIDIA GPU (`cuda`), Apple GPU (`mps`), or CPU.

## 5.5 Two levels of captions

### Image-level
The whole image goes to the model with the image prompt. The output is stored in the record's `captions` list.

### Object-level (the core of Week 2)
For each object:
1. Take its **bounding box**.
2. Enlarge it by **15% on each side** so the model sees a bit of context (what it's lying on).
3. **Crop** that region out of the image. If the crop is tiny (<32 px), enlarge it.
4. Send the crop to the model with the **object prompt**.
5. Save the caption with the object's `object_id` **and the exact crop rectangle used**.

Because the caption is stored under the object's `object_id`, it is **automatically linked to that object's bounding box and segmentation mask**. This is the "associate captions with boxes/masks" requirement.

**Which objects get captions?**
- **Tiny** objects are skipped (a 10-pixel cigarette crop is useless; the guidelines say tiny objects don't need object captions).
- At most **8 objects per image, largest first**, to keep dense ZeroWaste frames manageable.
- Pilot: 205 objects → 22 tiny skipped, 8 over the limit → **175 objects captioned**.

**Spatial relationships:** each object's location (e.g. "bottom-left") comes from its box. It is put into the prompt ("It lies in the bottom left part of the full photo") and into the template captions. Relations between objects ("next to", "on top of") are planned for Week 3 instructions.

## 5.6 The prompts (and why they look like this)

| Prompt | For | Text (short) |
|---|---|---|
| `blip_image` | BLIP-base | "a photo of" (prefix, then the model continues) |
| `blip2_image` | BLIP-2 | empty (BLIP-2 captions best without a prompt) |
| `chat_image` | Qwen/LLaVA | "You are annotating images for a waste-sorting dataset… name the waste items, material, physical state, where they lie… only what you clearly see, no brands." |
| `chat_image_short` | SmolVLM | "Describe this image in one sentence. Mention each piece of rubbish, what it is made of, and the surface it lies on." |
| `chat_object_json` | Qwen/LLaVA | Gives the label as a hint, asks for **JSON**: `{"caption": "...", "condition": [...]}` using only the 13 allowed condition words |
| `chat_object` | SmolVLM | "Describe the {label} in this photo in a few words, including its colour and condition." |

**Important design decision: object prompts include the known label.** We already *know* the box contains, e.g., a "clear plastic bottle". So we tell the model, and ask it to add what we *don't* know: **condition, colour, context**. This gives more accurate captions.
**Trade-off:** we then can't use object captions to test whether the model *recognises* objects by itself. (Image prompts contain **no** hints, so those are a fair test.)

**Why two prompt versions?** The tiny SmolVLM model **copied example words** from the long prompt: it said "crushed, torn, dirty" for every image because the prompt listed those words. This is called **prompt leakage**. So it got a shorter prompt. It also couldn't produce valid JSON (it copied the template text), so it gets a plain-text object prompt.

## 5.7 Cleaning the model output

- **JSON parsing:** if a model returns valid JSON, we read the caption and the conditions from it. If not, we fall back to the plain text.
- **Condition extraction:** a word list maps caption words to conditions ("flattened" → crushed, "dirty"/"greasy" → soiled, "ripped" → torn…). It is **negation-aware**: "not fully open" does *not* count as open.
- **Proposed, not applied:** extracted conditions go to `proposed.condition`. A human must accept them in the app before they become the official label.
- **Sentence cleanup:** small models ramble until they hit the word limit and get cut off mid-sentence. We keep at most 2 full sentences (image) or 1 (object). The full original output is always kept in `raw_text`.
- **BLIP fixes:** BLIP printed "close - up" instead of "close-up" (fixed), and repeated itself in loops ("a bag containing a bag containing…"). Fixed with `no_repeat_ngram_size=3`, which forbids repeating any 3-word sequence.

## 5.8 The template baseline (captions with no AI)

A third "model" called **`template`** writes captions **only from the labels**:
- Image: *"Four waste items indoors: two clear plastic bottles, a plastic bottle cap and a single-use carrier bag."*
- Object: *"plastic film in the top right of the image"*

Why it is useful:
1. It shows what a **complete** caption should mention (a checklist for reviewers).
2. It is a **baseline**: AI must beat it on things labels don't contain (condition, colour, context).
3. It **tests our automatic checker**: the template only says true things, so any warning the checker raises on it is a checker bug. After fixes: **0 false warnings on all 60 images.**

(For COCO, the template says "item" and not "waste item", because those objects may still be in use.)

## 5.9 Where the captions are stored

- `data/captions/<model>.jsonl`: raw captions, one per line, per model. **Resumable**: if the script stops, rerunning skips captions that already exist.
- Each caption records: which model, the exact prompt, settings, time taken, date. This is called **provenance**: you can always trace where a caption came from.
- `data/reviews/reviews.jsonl`: every human action from the app (approve, edit…), with reviewer name and time. It is only ever appended to, never overwritten.
- `10_merge_captions.py` combines the dataset + raw captions + reviews into the final records. It rebuilds from scratch every time, so it is always safe to rerun.
- `data/processed/captions.csv`: all captions as a spreadsheet.

**Each caption has an ID** like `taco_01113_o000__smolvlm-256m__chat_object__0`, meaning: object `taco_01113_o000`, model `smolvlm-256m`, prompt `chat_object`, candidate number `0`.

## 5.10 Schema v1.1

The schema was upgraded from v1.0 to v1.1 (old records still valid):
- Objects now have **`caption_candidates`** (AI captions for that object).
- Each caption has: `model`, `hf_id`, `prompt_id`, `prompt`, `clip_score`, `status` (pending/approved/edited/rejected), `review_score` (1–5), `hallucinated_objects`, `reviewer`, `original_text` (the AI text before a human edit), `proposed` conditions, `auto_checks`.
- The validator now also checks that **every caption is linked to the correct image/object**.
- Result: **2,410 records, 705 captions, 0 problems.**

## 5.11 Quality control: CLIPScore + automatic checks

### CLIPScore (`09_clip_score.py`)
Every caption gets a CLIPScore. Object captions are scored against **the same crop the model saw**.

### Automatic checks (`captionlib.py`)
Word lists turn caption words into taxonomy values, so each caption can be compared with the true labels:

| Check | What it detects | Example |
|---|---|---|
| material mentioned / recall | Does the caption say the material(s)? | labels say plastic; caption says "plastic" ✅ |
| material conflict | Names a material, but not the right one | label glass, caption says "plastic bottle" ⚠️ |
| category recall | What share of the labelled object types is named? | 2 of 4 types named = 0.5 |
| **unsupported categories** | Names an object type that is **not in the labels**: possible hallucination | caption mentions "straws", none labelled ⚠️ |
| **people / animals** | People/animals are never labelled in waste data, so they are likely hallucinated | "a dead animal on the sand" ⚠️ |
| brand / text | Quotes a brand or label text (not allowed) | "has the word 'Lucky' on it" ⚠️ |
| mentions condition | Uses a condition word | "crumpled" ✅ |

Tricky cases we fixed: "we **can** see" is not a can; "food can" is not food waste; "bottle cap" is a cap, not a bottle; a paper cup is "composite"; plurals ("films"); negations. For ZeroWaste images (with unlabelled paper everywhere), the "unsupported" check is relaxed so correct captions are not falsely flagged.

**These checks are signals, not truth.** They tell the human reviewer *where to look first*.

---

# Part 6: The Streamlit curation app

Run it with: `streamlit run app/streamlit_app.py` → it opens in the browser.

### Sidebar
- **Reviewer name**, saved with every action.
- **Filters:** split, source, category, review status, "has pending captions", auto-check warnings (unsupported object, brand), minimum CLIPScore.
- **Navigation:** Prev / Next buttons, image dropdown.

### Tab 1: 🖼️ Image captions
- The image with **boxes** (and optional **masks**) drawn on it.
- Metadata: source, split, size, object counts, environment, quality flags, license, source link.
- **Scene form:** environment, "items discarded?", free-text scene description, image review status, notes.
- **Side-by-side model comparison:** two or three models in columns (default BLIP-2 vs Qwen2.5-VL; in the pilot, BLIP-base vs SmolVLM).
- For each caption: text, status icon (⏳ pending, ✅ approved, ✏️ edited, ❌ rejected), CLIPScore, warning badges, and a review panel to **edit, give a 1–5 score, tag hallucinated objects, Approve / Save edit / Reject**.
- **🔁 Regenerate** button per model.
- **Write your own caption** box.

### Tab 2: 📦 Objects
- Pick an object; it is highlighted on the image, and the **exact crop the AI saw** is shown.
- **Structured fields:** category, material, condition (multi-select), sorting action, and the count of that category in the image.
- **"Accept VLM conditions"** button: one click to accept what the AI proposed.
- That object's caption candidates with the same approve/edit/reject controls.

### Tab 3: ⚙️ Generate
- Choose **any model**, image or object level, how many candidates (1–5), a prompt template, **edit the prompt freely**, preview the final prompt, press Generate. The result is saved and scored with CLIPScore immediately.

### Tab 4: 📊 Statistics
- Number of images, images with captions, total captions, reviewed count, **approval rate**.
- Charts: objects per category (**class distribution**), review status, **captions per category** (all vs approved), mean CLIPScore per model.

### Tab 5: ⬇️ Export
- Choose statuses (default: approved + edited) and splits.
- Download **JSONL** (image + object captions), **CSV** of image captions, **CSV** of object captions.

### Design choice to mention
The app **never edits the dataset directly**. It only appends actions to a log file (`reviews.jsonl`). Benefits: full **audit trail** (who changed what, when), nothing is ever lost, and it allows measuring agreement between two reviewers later.

### Required features checklist (from the project brief)

| Required | Done? |
|---|---|
| Upload/select images, show metadata | ✅ select + filter + metadata (uploading *new* images: planned Week 4) |
| Select model, enter prompt template | ✅ |
| Generate one or multiple candidates | ✅ |
| Edit / approve / reject / regenerate | ✅ |
| Structured fields: object, material, condition, count, scene context | ✅ |
| Quality score / review score, filtering | ✅ |
| Side-by-side BLIP-2 vs Qwen2.5-VL | ✅ (shows pilot models until the GPU run exists) |
| Export JSON/JSONL and CSV | ✅ |
| Statistics: images, class distribution, approval rate, captions per class | ✅ |

**Tested** automatically with Streamlit's testing tool (`AppTest`): loading, navigation, approving, saving fields, export, live generation. All ran without errors. Test reviews were deleted afterwards, so **no real human reviews exist yet**.

---

# Part 7: Results and what we learned (pilot, 60 images)

Full report: `data/reports/captions_week2.md`. Example picture: `data/reports/caption_examples.jpg` (open it to show examples).

## 7.1 Image-level results

| metric | BLIP-base | SmolVLM-256M | template (labels) |
|---|---:|---:|---:|
| average words | 7.0 | 24.6 | 13.1 |
| CLIPScore | **0.747** | 0.745 | 0.703 |
| mentions a material | 23% | 45% | 68% |
| **category recall** (share of labelled object types named) | **0.06** | **0.22** | 1.00 |
| unsupported category (possible hallucination) | 6.7% | 3.3% | 0% |
| mentions people/animals (TACO + ZeroWaste) | **13.3%** | 2.2% | 0% |
| mentions a condition | 6.7% | 6.7% | 0% |

## 7.2 Object-level results (175 objects)

| metric | BLIP-base | SmolVLM-256M | template |
|---|---:|---:|---:|
| average words | 6.5 | 14.9 | 9.4 |
| CLIPScore | 0.655 | **0.674** | 0.611 |
| material match | 20% | 96%\* | 100% |
| category match | 13% | 99%\* | 100% |
| mentions a condition | 0.6% | 17.7% | 0% |

\* **Inflated**: SmolVLM's object prompt contains the label, so of course it repeats it. Example: *"The clear plastic bottle is red with a green label."* It repeated "clear" from the hint and then contradicted itself.

Conditions SmolVLM proposed: intact 23, torn 5, soiled 2, crumpled 1.

## 7.3 The five key findings (memorise these)

1. **AI models leave small waste out of whole-image captions.** In beach/grass photos with small litter, both models describe only the background: *"the sand at the beach"*, *"a leaf litter on the ground"*. They name only 6–22% of the labelled object types.
   → **This justifies object-level captioning on crops.**

2. **CLIPScore is NOT a good quality filter for waste captions.** The template captions (100% correct, built from true labels) scored *lower* than BLIP's generic captions. BLIP and SmolVLM got almost the same CLIPScore although SmolVLM named 3× more of the real objects. CLIP rewards describing the **big obvious scene** ("a plate of food with a fork and knife") rather than the small waste item.
   → **Use CLIPScore only to sort/rank captions, not as a pass/fail threshold. Structured checks + human review are needed.**

3. **Hallucinations are often non-waste objects.** *"The body of a man found in the jungle"* (actually plastic bags in bushes), *"a dead animal lying on the sand"*, *"a seat with a stuffed animal on it"*, *"a beach with a surf board"*. The waste-category check missed them, so we added the people/animals check. BLIP did this in 13% of TACO/ZeroWaste captions.

4. **Small models fall into stock phrases and copy prompts.** SmolVLM wrote *"The box is open, revealing a glimpse of…"* for 20+ cardboard pieces (mode collapse), and copied example words from a long prompt (prompt leakage).

5. **Brand text appears** (*"has the word 'Lucky' on the front"*), which the guidelines forbid. It is now flagged automatically.

Also:
- **ZeroWaste:** "a pile of paper" is actually *correct*, because those frames are full of unlabelled newspaper. That's why checks are relaxed there.
- **COCO works as intended as a contrast set:** models describe meals and tables, not waste.

---

# Part 8: Limitations, and what comes next

## Limitations (say these yourself, before the professor does)
1. **The target models (BLIP-2, Qwen2.5-VL) have not been run yet.** They need a GPU. The pipeline is ready (`run_week2_gpu.sh`).
2. **The pilot is small** (60 images) and uses weak stand-in models.
3. **Object-level agreement numbers are inflated** by the label hint in the prompt. Plan: an un-hinted run on the test split for evaluation.
4. **Automatic checks are keyword-based.** They miss paraphrases and some hallucinations. We'll measure how accurate they are against human labels in Week 3.
5. **Few conditions detected** so far (18% of objects). Qwen with a JSON prompt should do better.
6. **One reviewer** so far. A second reviewer on ~10% of the images would allow measuring agreement (Cohen's kappa).
7. **License:** ZeroWaste is non-commercial, so the final dataset will be **CC BY-NC 4.0**.
8. Rare categories (food_waste 8, hazardous 2) are too rare to evaluate separately.

## Next steps
- **Week 3:** run BLIP-2 + Qwen2.5-VL on a GPU; human verification in the app (flagged captions first); apply the sorting override rules; generate **grounded sorting instructions** per object, e.g. *"Pick up the crushed plastic bottle at the bottom-left and place it in the plastic recycling bin"*, each linked to one object box/mask.
- **Week 4:** final JSON dataset; evaluate a baseline VLM on caption quality (BLEU, CIDEr, CLIPScore), **hallucination rate**, **grounding accuracy**; add image upload to the app.
- **Week 5:** error analysis, dataset card, guidelines, GitHub repo, final report, demo video.

## What to ask the professor
1. Can I get **GPU access** (≥16 GB VRAM)?
2. Is it OK to give the object label as a **hint** in object prompts, with an un-hinted run on the test split for evaluation?
3. Given finding 2, should CLIPScore be used **only for ranking, not filtering**?

---

# Part 9: Questions your professor may ask, with answers

**Q: What is the main contribution of this project?**
A verified waste image–caption dataset with structured, object-level descriptions (category, material, condition, location, sorting action) linked to boxes and masks, plus the tools to make it: the generation pipeline, automatic quality checks, and a curation app.

**Q: Why not just use the class labels? Why captions?**
Labels say "Clear plastic bottle". A caption says "crushed clear plastic bottle half-buried in sand at the bottom-left". That adds condition, context, and location in natural language, which is what vision-language models and robots that take language instructions (VLA models) need for training and evaluation.

**Q: Why these three datasets?**
TACO gives detailed object types in real outdoor scenes. ZeroWaste gives extreme clutter from a real recycling plant. COCO gives non-waste bottles/cups as a contrast set to test false "waste" claims.

**Q: Why is COCO in a waste dataset?**
As **hard negatives**: a good model should not call a wine glass on a dinner table "waste". Those records have `is_discarded: unknown` and are reported separately.

**Q: How did you avoid data leakage?**
ZeroWaste keeps its official split by video, so frames from one video never appear in both train and test. Near-duplicate images were removed with perceptual hashing.

**Q: Why separate category / material / condition?**
They vary independently: a bottle can be plastic or glass, intact or crushed. Separate fields allow structured captions ("crushed plastic bottle") and allow measuring errors per attribute.

**Q: Why caption cropped objects instead of the whole image?**
Our pilot showed whole-image captions miss most small waste (only 6–22% of labelled object types named). A crop forces the model to look at one object, and the caption is automatically linked to that object's box/mask (grounding).

**Q: Why add 15% padding to crops?**
So the model sees a bit of the surroundings (what the object lies on), which helps describe context and recognise the object.

**Q: Isn't giving the label in the prompt cheating?**
For *building* the dataset it's fine: the label is already verified, and we only want the model to add condition, colour, and context. For *evaluating* recognition it would be cheating, so evaluation will use un-hinted prompts, and image-level prompts never get hints.

**Q: What is CLIPScore and why is it not enough?**
It measures image–text similarity with the CLIP model, no reference needed. In our pilot, fully correct label-based captions scored lower than generic scene captions, because CLIP rewards describing the dominant scene, not small waste. So we combine it with structured checks and human review.

**Q: Why not BLEU / CIDEr now?**
They need human-written reference captions. Those will exist after Week 3 verification, so they are used in Week 4.

**Q: How do you detect hallucinations?**
Automatically: we compare object words in the caption with the labelled objects ("unsupported categories") and flag people/animals and brand text. Manually: reviewers tag hallucinated objects in the app. Week 3 will measure how accurate the automatic flags are.

**Q: Why didn't you run BLIP-2 and Qwen?**
Hardware: an 8 GB laptop with no GPU. Those models need about 8–17 GB of GPU memory and 15+ GB of disk. The code is model-agnostic, a ready script runs them on a GPU, and I ran a 60-image pilot with smaller models from the same two families to test everything.

**Q: Why two model types (BLIP and Qwen)?**
They represent two families: a **captioning** model (short, generic, can't follow instructions) and an **instruction-following chat** model (can be asked for material, condition, JSON). Comparing them shows whether instruction following helps structured waste captions.

**Q: What is the template baseline for?**
It writes captions only from the labels. It's a reference for completeness, a baseline for AI to beat on condition/context, and a test of the automatic checker (it got 0 false warnings).

**Q: How is the work reproducible?**
Fixed seed 42, a config file for all settings, every caption stores its model/prompt/settings, raw outputs kept, the merge step is idempotent (rerunning gives the same result), and the review log is append-only.

**Q: How will humans verify captions?**
In the Streamlit app: approve / edit / reject each caption, score it 1–5, tag hallucinated objects, accept or fix proposed conditions, correct category/material. Every action is logged with reviewer and time.

**Q: What does "grounded instruction" mean?**
A sorting instruction tied to one specific object's box/mask: "Pick up the crushed plastic bottle at the bottom-left and put it in the plastic bin". A robot then knows exactly which object is meant.

**Q: How big is the dataset?**
2,410 images, 9,391 objects; splits 1,712 / 355 / 343. So far 705 caption candidates on the 60-image pilot.

**Q: What surprised you?**
That CLIPScore preferred generic, wrong-focus captions over correct ones, and that hallucinations were often people or animals rather than wrong waste objects.

**Q: What's the license?**
Non-commercial (CC BY-NC 4.0), because ZeroWaste is non-commercial.

---

# Part 10: A 5-minute speech you can give

> "My project is CaptionGene: building a verified dataset of waste images with descriptive captions.
>
> **The problem:** recycling systems need AI that understands trash: what an object is, what it's made of, and its condition. General vision-language models like BLIP or Qwen produce weak captions for waste scenes. They miss small items, confuse materials, and hallucinate. Public waste datasets have boxes and labels, but no captions.
>
> **Week 1** was the dataset. I combined TACO (litter in the wild), ZeroWaste (a recycling-plant conveyor belt), and COCO bottles and cups as a non-waste contrast set. After cleaning and de-duplication I have 2,410 images with 9,391 labelled objects. I designed one taxonomy: every object gets a category, a material, a condition, a location, and a sorting action. I mapped all 67 original labels onto it and formalised everything as a JSON schema. The splits are 1,712 / 355 / 343, keeping ZeroWaste's video-level split to avoid leakage.
>
> **Week 2** was caption generation. I built a config-driven pipeline that writes captions at two levels: the whole image, and each object, using a padded crop around its box. Object captions are linked by ID to their box and mask. It supports BLIP-2, Qwen2.5-VL, and LLaVA. Every caption is scored with CLIPScore and with automatic checks against the labels: material and category agreement, objects that aren't labelled (possible hallucinations), people or animals, and brand names.
>
> My laptop can't run BLIP-2 or Qwen, so I ran a 60-image pilot with small models from the same two families. There's a ready script for the full GPU run.
>
> **Three findings:** first, whole-image captions miss most small waste, which supports object-level captioning. Second, CLIPScore is not a reliable filter here: captions built from the true labels scored lower than generic scene captions. Third, hallucinations are often people or animals, like 'the body of a man in the jungle' for plastic bags in bushes.
>
> I also built the **Streamlit curation app**: side-by-side model comparison, approve/edit/reject/regenerate, structured fields, custom prompts, statistics, and JSONL/CSV export. Every action is logged.
>
> **Next week:** run the real models on a GPU, do human verification, and generate grounded sorting instructions. I'd like to ask about GPU access."

---

# Part 11: How to demo it live

Open a terminal in the project folder:

```bash
cd ~/Desktop/CaptionGene
streamlit run app/streamlit_app.py
```

The browser opens. Suggested 3-minute demo:
1. **Sidebar:** type your name; point out that filters show "60 of 2410 images match" (the pilot).
2. **Image captions tab:** show the boxes on the image, the metadata, and the **two models side by side**. Point at a ⚠️ badge (people/animals or brand). Open "Review" on one caption, change a word, press **Approve** → it turns ✏️ edited.
3. **Objects tab:** select an object → it's highlighted, and the crop the AI saw is shown. Show the material/condition fields and "Accept VLM conditions".
4. **Generate tab:** choose `smolvlm-256m`, press **Generate** (takes a few seconds the first time while the model loads), and the new caption appears with its CLIPScore.
5. **Statistics tab:** class distribution, approval rate.
6. **Export tab:** the download buttons.

⚠️ Any approve/edit you do in the demo is **saved** to `data/reviews/reviews.jsonl`. If it was just a demo, delete that file afterwards and run `python scripts/10_merge_captions.py`.

Also good to open: `data/reports/caption_examples.jpg` (examples) and `data/reports/captions_week2.md` (all numbers).

---

# Part 12: A map of every file

```
CaptionGene/
├── README.md                     overview + how to run
├── PROJECT_EXPLAINED.md          ← this file
├── WEEK1_UPDATES.md              Week 1 report (dataset)
├── WEEK2_UPDATES.md              Week 2 report (captioning)
├── requirements.txt              Python packages needed
│
├── configs/
│   ├── taxonomy.yaml             categories, materials, conditions, bins, label mappings
│   └── captioning.yaml           models, prompts, generation settings  (Week 2)
├── schema/annotation_schema.json rules for a valid record (v1.1)
├── docs/annotation_guidelines.md rules for human annotators (v0.2)
│
├── scripts/
│   ├── common.py                 shared paths and helpers
│   ├── 01_download_taco.py       ┐
│   ├── 02_download_zerowaste.py  │ Week 1: get the images
│   ├── 03_download_coco.py       ┘
│   ├── 04_build_manifest.py      Week 1: clean + convert to schema
│   ├── 05_make_splits.py         Week 1: train/val/test
│   ├── 06_dataset_stats.py       Week 1: statistics
│   ├── 07_validate_schema.py     checks everything is valid
│   ├── vlm.py                    Week 2: model wrappers + CLIPScore
│   ├── captionlib.py             Week 2: crops, prompts, parsing, checks, merge
│   ├── 08_generate_captions.py   Week 2: run the models
│   ├── 09_clip_score.py          Week 2: score captions
│   ├── 10_merge_captions.py      Week 2: put captions + reviews into the dataset
│   ├── 11_caption_report.py      Week 2: report + example image
│   └── run_week2_gpu.sh          Week 2: full run on a GPU
│
├── app/streamlit_app.py          the curation web app
│
└── data/
    ├── images/{taco,zerowaste,coco}/   the 2,410 images
    ├── processed/                manifest_all.jsonl (THE dataset), train/val/test.jsonl,
    │                             objects.csv, captions.csv
    ├── captions/                 raw AI captions per model + pilot_ids.txt
    ├── reviews/                  reviews.jsonl (human actions; created on first review)
    └── reports/                  stats.md, cleaning_report.json, sample_grid.jpg (W1),
                                  captions_week2.md/json, caption_examples.jpg (W2)
```

---

# Part 13: Glossary

| Term | Meaning |
|---|---|
| **VLM** | Vision-Language Model: AI that reads images and writes/understands text |
| **VLA** | Vision-Language-Action model: AI that also acts (e.g. controls a robot) from images + language |
| **Caption** | A text description of an image or object |
| **Prompt** | The instruction text given to a chat model |
| **Hallucination** | The AI describes something not in the image |
| **Bounding box (bbox)** | Rectangle around an object: [x, y, width, height] |
| **Segmentation mask** | Exact polygon outline of an object |
| **Grounding** | Linking text to a specific region (box/mask) in the image |
| **Object-level / image-level** | Caption for one object / for the whole image |
| **Crop** | A cut-out part of an image (here: around a box, +15%) |
| **Taxonomy** | Fixed list of allowed categories/values |
| **Schema** | Formal rules for what a valid data record looks like |
| **JSON / JSONL / CSV** | Data formats: structured text / one JSON per line / spreadsheet table |
| **Train / val / test split** | Data for learning / tuning / final evaluation |
| **Data leakage** | Test data (or near-copies) sneaking into training, giving falsely high scores |
| **Stratification** | Splitting so every class appears in each split in similar proportions |
| **Perceptual hash (pHash)** | Image fingerprint; similar images have similar hashes (finds near-duplicates) |
| **MD5 hash** | File fingerprint; identical files have identical hashes (finds exact duplicates) |
| **EXIF** | Metadata in phone photos, including orientation |
| **Occlusion** | An object partly hidden behind another |
| **Clutter level** | How many objects are in the image (single/sparse/moderate/dense) |
| **MRF** | Material Recovery Facility (recycling plant) |
| **Contrast set / hard negatives** | Examples that look similar but are not the target (COCO bottles in use) |
| **CLIP / CLIPScore** | Image-text matching model / a caption quality score based on it (0–2.5) |
| **BLEU / CIDEr** | Caption metrics comparing to human reference captions (word overlap) |
| **Cosine similarity** | How similar two vectors (lists of numbers) are, from -1 to 1 |
| **Parameters (7B, 256M)** | The number of learned numbers inside a model; bigger is usually smarter but heavier |
| **GPU / VRAM** | Graphics card that runs AI fast / its memory |
| **CUDA / MPS** | NVIDIA GPU platform / Apple-silicon GPU platform |
| **Hugging Face / transformers / PyTorch** | Model hub / library to run models / neural-network library |
| **Greedy decoding** | The model always picks the most likely next word (same output every time) |
| **Sampling / temperature** | Random word choice for varied outputs; higher temperature = more random |
| **Token** | A piece of a word; models count length in tokens (60 tokens ≈ 45 words) |
| **Prompt leakage** | The model copies words from the prompt instead of describing the image |
| **Mode collapse** | The model repeats the same stock phrase for many inputs |
| **Baseline** | A simple reference method that better methods should beat |
| **Provenance** | Record of where each piece of data came from (model, prompt, time) |
| **Idempotent** | Running it again gives the same result (safe to rerun) |
| **Append-only log** | A file that only gets new lines added, never edited (audit trail) |
| **Cohen's kappa (κ)** | Measure of agreement between two annotators, corrected for chance |
| **Streamlit** | Python library for quick web apps |
| **Pilot** | A small trial run to test the pipeline before the full run |
| **Stand-in model** | A small model used in place of the target model because of hardware limits |
