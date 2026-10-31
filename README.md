# CaptionGene: Vision-Language Caption Generation for Waste Images

A verified waste image–caption–instruction dataset, built by generating captions with pretrained VLMs (BLIP-2, LLaVA, Qwen2.5-VL) and refining them through quality checks and human review in a Streamlit curation app.

## Repository layout

```
configs/taxonomy.yaml          unified waste taxonomy + source-label mappings
configs/captioning.yaml        VLM registry, prompt templates, generation + crop settings
schema/annotation_schema.json  JSON Schema for one image record
docs/annotation_guidelines.md  caption / verification / instruction rules
scripts/
  common.py                    shared paths + image/annotation helpers
  01_download_taco.py          TACO: 1,500 images (EXIF-fixed, resized, boxes rescaled)
  02_download_zerowaste.py     ZeroWaste-f: ~600-frame subset via HTTP range requests
  03_download_coco.py          COCO val2017: bottle / cup / wine-glass subset
  04_build_manifest.py         cleaning, dedup, taxonomy mapping -> unified schema
  05_make_splits.py            70/15/15 stratified splits + objects.csv
  06_dataset_stats.py          statistics tables + sample grid
  07_validate_schema.py        schema + bbox + file-existence + caption-link validation
  vlm.py                       BLIP / chat-VLM wrappers + CLIPScore
  captionlib.py                crops, prompts, output parsing, template captions, auto checks, merge
  08_generate_captions.py      image- and object-level caption candidates (resumable)
  09_clip_score.py             reference-free CLIPScore for every candidate
  10_merge_captions.py         candidates + review log -> annotation records, captions.csv
  11_caption_report.py         caption quality report + example sheet
  run_week2_gpu.sh             full run with BLIP-2 + Qwen2.5-VL on a CUDA GPU
  make_colab_bundle.sh         zip code + images for the Colab run
  import_colab_results.sh      bring Colab captions back, re-merge, report, validate
notebooks/week2_colab.ipynb    Google Colab notebook for the GPU run
app/streamlit_app.py           curation app: review, edit, generate, statistics, export
data/
  raw/        source annotation files (TACO json, ZeroWaste labels, COCO instances)
  interim/    per-source records after download
  images/     processed JPEGs (longest side <= 1280 px)
  captions/   raw caption candidates, one file per model (+ pilot_ids.txt)
  reviews/    reviews.jsonl - append-only log of reviewer actions from the app
  processed/  manifest_all.jsonl, train/val/test.jsonl, objects.csv, captions.csv
  reports/    cleaning + dataset stats (week 1), captions_week2.md/json, caption_examples.jpg
WEEK1_UPDATES.md               progress report for week 1
WEEK2_UPDATES.md               progress report for week 2
```

## Reproduce

```bash
pip install -r requirements.txt
cd scripts
python 01_download_taco.py
python 02_download_zerowaste.py      # --train 420 --val 90 --test 90
python 03_download_coco.py           # --n 400
python 04_build_manifest.py
python 05_make_splits.py
python 06_dataset_stats.py
python 07_validate_schema.py

# Week 2 - local pilot (60 val images, small models, CPU / Apple silicon)
python 08_generate_captions.py --models template blip-base smolvlm-256m --subset pilot
python 09_clip_score.py
python 10_merge_captions.py
python 11_caption_report.py
python 07_validate_schema.py

# Week 2 - full run with the target models (CUDA GPU)
bash run_week2_gpu.sh

# ... or on Google Colab (no local GPU):
bash make_colab_bundle.sh                  # -> ~/CaptionGene_colab.zip, upload to MyDrive/CaptionGene/
#    open notebooks/week2_colab.ipynb in Colab (T4 GPU), run all cells, download captions_out.zip
bash import_colab_results.sh ~/Downloads/captions_out.zip

# Curation app (from the repo root)
streamlit run app/streamlit_app.py
```

All randomness uses seed 42.

## Licenses of sources

- TACO: CC BY 4.0
- ZeroWaste: CC BY-NC 4.0 (non-commercial)
- COCO: CC BY 4.0 for annotations; each image keeps its own Flickr license

The combined dataset therefore inherits **non-commercial** terms wherever ZeroWaste images are included.
