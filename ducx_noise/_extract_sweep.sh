#!/bin/bash
# Re-extract the confirmatory sweep500 logs with the temp-0 LLM extractor.
# Assumes the Qwen2.5 extractor vLLM is serving on $PORT.
set -eo pipefail
export DS=/datasets/omni_pretraining
export DUCK=/project/6101776/xzhan576/DUCK
DUCK_PY=$DS/.venvs/DUCK/bin/python
PORT=${PORT:-8011}
export EXTRACTOR_BASE_URL="http://localhost:$PORT/v1"
export EXTRACTOR_MODEL="qwen2.5-7b-instruct"
export EXTRACTOR_API_KEY=EMPTY
CACHE_DIR=$DUCK/extraction_cache
OUT_DIR=$DUCK/logs/sweep500_extract
RESULTS_CSV=$OUT_DIR/results_extract.csv
mkdir -p "$OUT_DIR"; rm -f "$RESULTS_CSV"
cd "$DUCK"
curl -sf "$EXTRACTOR_BASE_URL/models" >/dev/null 2>&1 || { echo "extractor not up"; exit 2; }
for c in base distractor_5 distractor_10 distractor_20 distractor_50; do
  f=$(ls -t logs/sweep500_${c}/sweep500_${c}_*.json 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "skip $c (no log)"; continue; }
  echo "=== $c <- $(basename "$f") ==="
  $DUCK_PY -m ducx_noise.extract recompute \
    --run-log "$f" --label "$c" --seed 0 \
    --cache-dir "$CACHE_DIR" --out "$RESULTS_CSV" \
    --records-out "$OUT_DIR/records_${c}.jsonl" \
    --base-url "$EXTRACTOR_BASE_URL" --model "$EXTRACTOR_MODEL"
done
echo "=== aggregate ==="
$DUCK_PY -m ducx_noise.extract aggregate --in "$RESULTS_CSV" \
  --summary-out "$OUT_DIR/summary_extract.md"
echo "SWEEP_EXTRACT_DONE"
