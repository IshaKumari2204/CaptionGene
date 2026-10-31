#!/usr/bin/env bash
# Pack what the Colab GPU run needs into one zip (code, configs, schema, processed records, images).
# Pilot captions, raw downloads and reports stay on the Mac.
#
#   bash scripts/make_colab_bundle.sh            # -> ~/CaptionGene_colab.zip (~900 MB; outside the iCloud-synced Desktop)
#
# Then upload the zip to Google Drive under MyDrive/CaptionGene/ and open notebooks/week2_colab.ipynb in Colab.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/CaptionGene_colab.zip}"
rm -f "$OUT"
# images may be iCloud placeholders ("dataless"); zip reads them, which downloads them first
zip -r -q "$OUT" scripts configs schema notebooks requirements.txt \
    data/processed data/images -x "*/__pycache__/*" "*.DS_Store"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), $(unzip -l "$OUT" | grep -c '\.jpg$') images"
