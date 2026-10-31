#!/usr/bin/env bash
# Bring the Colab caption run back to the Mac and rebuild the merged records + report locally (no GPU needed).
#
#   bash scripts/import_colab_results.sh ~/Downloads/captions_out.zip
#
# The zip holds data/captions/<model>.jsonl (with CLIPScores) as written by notebooks/week2_colab.ipynb.
# The pilot stand-in captions (blip-base, smolvlm-256m) are moved to data/captions/pilot/ so they no longer
# appear in the app or report; move them back to restore the pilot.
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP="${1:?usage: import_colab_results.sh <captions_out.zip>}"
mkdir -p data/captions/pilot
for m in blip-base smolvlm-256m; do
    [ -f "data/captions/$m.jsonl" ] && mv "data/captions/$m.jsonl" data/captions/pilot/
done
[ -f data/captions/pilot_ids.txt ] && mv data/captions/pilot_ids.txt data/captions/pilot/
unzip -o -j "$ZIP" "*.jsonl" -d data/captions
ls -l data/captions/*.jsonl
cd scripts
python 10_merge_captions.py
python 11_caption_report.py > /dev/null
python 07_validate_schema.py
echo "done: report in data/reports/captions_week2.md; start the app with: streamlit run app/streamlit_app.py"
