#!/usr/bin/env bash
# Full Week 2 caption run with the target models on a CUDA GPU (Colab T4/L4/A100, lab server).
# BLIP-2 OPT-2.7B fp16 needs ~8 GB VRAM, Qwen2.5-VL-7B bf16 ~17 GB (use qwen2.5-vl-3b on a 16 GB T4).
#
#   bash scripts/run_week2_gpu.sh                 # all 2,410 images
#   SUBSET=split:test bash scripts/run_week2_gpu.sh
#   QWEN=qwen2.5-vl-3b bash scripts/run_week2_gpu.sh
#   QWEN_BS=4 BLIP2_BS=8 bash scripts/run_week2_gpu.sh     # smaller batches if CUDA runs out of memory
#
# Re-running is safe: 08_generate_captions.py skips captions that already exist (resume after a disconnect).
set -euo pipefail
cd "$(dirname "$0")"
SUBSET="${SUBSET:-all}"
QWEN="${QWEN:-qwen2.5-vl-7b}"
BLIP2_BS="${BLIP2_BS:-16}"
QWEN_BS="${QWEN_BS:-8}"

python 08_generate_captions.py --models template --subset "$SUBSET"
python 08_generate_captions.py --models blip2 --subset "$SUBSET" --batch-size "$BLIP2_BS"
python 08_generate_captions.py --models "$QWEN" --subset "$SUBSET" --batch-size "$QWEN_BS"
python 09_clip_score.py
python 10_merge_captions.py
python 11_caption_report.py
python 07_validate_schema.py
