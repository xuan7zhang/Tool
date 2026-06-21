#!/bin/bash
# Task 2 -- re-extract every existing rollout with the temp-0 LLM extractor and
# rebuild the noisy task_acc table (old vs new). Touches ONLY answer extraction;
# sel_acc / entropy / calls are unchanged.
#
# Extractor model = Qwen2.5-7B-Instruct (text-only, a DIFFERENT model from the
# Qwen3-VL-8B under test -> no self-grading), served on a spare GPU via vLLM.
#
# Usage (inside a GPU allocation):
#   srun --jobid=3969910 --overlap --ntasks=1 --pty bash \
#     ducx_noise/run_extract_recompute.sh
set -eo pipefail

export DS=/datasets/omni_pretraining
export DUCK=${DUCK:-/project/6101776/xzhan576/DUCK}
export HF_HOME=${HF_HOME:-$DS/hf_cache}          # where Qwen2.5-7B-Instruct lives
VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python

EXTRACT_MODEL_ID=${EXTRACT_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
EXTRACT_SERVED=${EXTRACT_SERVED:-qwen2.5-7b-instruct}
PORT=${PORT:-8011}                                # off the eval default (8000/8001)
EX_GPU=${EX_GPU:-0}

CACHE_DIR=${CACHE_DIR:-$DUCK/extraction_cache}
OUT_DIR=${OUT_DIR:-$DUCK/logs/extract_recompute}
RESULTS_CSV=$OUT_DIR/results_extract.csv
mkdir -p "$CACHE_DIR" "$OUT_DIR"
rm -f "$RESULTS_CSV"

cd "$DUCK"

# ---- 1. serve the extractor model (skip if EXTRACTOR_BASE_URL already set) ----
VLLM_PID=""
if [ -z "$EXTRACTOR_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$EX_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$EXTRACT_MODEL_ID" --served-model-name "$EXTRACT_SERVED" --port $PORT \
    --gpu-memory-utilization 0.55 --max-model-len 16384 \
    > "$OUT_DIR/vllm_extract_p${PORT}.log" 2>&1 &
  VLLM_PID=$!
  trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
  echo "Waiting for extractor vLLM on :$PORT ..."
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "extractor ready"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -40 "$OUT_DIR/vllm_extract_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export EXTRACTOR_BASE_URL="http://localhost:$PORT/v1"
fi
export EXTRACTOR_MODEL="$EXTRACT_SERVED"
export EXTRACTOR_API_KEY="${EXTRACTOR_API_KEY:-EMPTY}"

# ---- 2. re-extract every condition's latest run log ----
for d in "$DUCK"/logs/noise_*seed*; do
  [ -d "$d" ] || continue
  label_full=$(basename "$d")                     # e.g. noise_distractor_5_seed0
  # split trailing _seed<N>
  seed=$(echo "$label_full" | sed -E 's/.*_seed([0-9]+)$/\1/')
  label=$(echo "$label_full" | sed -E 's/^noise_//; s/_seed[0-9]+$//')
  run_log=$(ls -t "$d/${label_full}_"*.json 2>/dev/null | head -1 || true)
  [ -n "$run_log" ] || { echo "skip $label_full (no run log)"; continue; }
  echo "=== $label seed=$seed  <- $(basename "$run_log") ==="
  $DUCK_PY -m ducx_noise.extract recompute \
    --run-log "$run_log" --label "$label" --seed "$seed" \
    --cache-dir "$CACHE_DIR" --out "$RESULTS_CSV" \
    --records-out "$OUT_DIR/records_${label_full}.jsonl" \
    --base-url "$EXTRACTOR_BASE_URL" --model "$EXTRACTOR_MODEL"
done

# ---- 3. aggregate old-vs-new task_acc table ----
echo
echo "=== Old vs New task_acc (mean +/- std over seeds) ==="
$DUCK_PY -m ducx_noise.extract aggregate --in "$RESULTS_CSV" \
  --summary-out "$OUT_DIR/summary_extract.md" --summary-csv "$OUT_DIR/summary_extract.csv"
echo
echo "Per-run rows: $RESULTS_CSV"
echo "Summary:      $OUT_DIR/summary_extract.md"
echo "Cache:        $CACHE_DIR"
