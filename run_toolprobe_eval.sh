#!/bin/bash
# Tool-probe model-compare backfill: run ONE backbone over the base (no-noise)
# question set, saving full traces so per-tool selection/call/success/correctness
# can be aggregated later. Same query set + same judge as the existing 8B run, so
# the only variable is the backbone (strong-vs-weak per-tool gap study).
#
# Usage (inside a 4-GPU allocation), weak end (recommended start):
#   SERVED_NAME=qwen3-vl-2b MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
#   MAX_CASES=500 bash run_toolprobe_eval.sh
#
# Strong end (32B, needs tensor-parallel across the 4 L40s):
#   SERVED_NAME=qwen3-vl-32b MODEL_ID=Qwen/Qwen3-VL-32B-Instruct TP=4 \
#   VLLM_GPU=0,1,2,3 TOOL_GPU=0 MAX_CASES=500 bash run_toolprobe_eval.sh
set -eo pipefail

export DS=/datasets/omni_pretraining
# Reuse the SAME hf cache that already holds the 8B weights (so 8B is not
# re-downloaded; new backbones download here too).
export HF_HOME=${HF_HOME:-/scratch/$USER/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
mkdir -p "$HF_HOME" "$MEDRAX_WEIGHTS" "$MEDRAX_TEMP" "$DUCK/logs"

VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=${SERVED_NAME:-qwen3-vl-2b}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
PORT=${PORT:-8000}
VLLM_GPU=${VLLM_GPU:-0}      # serve GPU(s); comma list when TP>1
TOOL_GPU=${TOOL_GPU:-1}      # GPU for the medrax tools
TP=${TP:-1}
MAX_CASES=${MAX_CASES:-500}
CONFIG=${CONFIG:-ducx_noise/configs/base.yaml}   # base = identity (no injected noise)
OUT=${OUT:-$DUCK/logs/toolprobe_${SERVED_NAME}}
mkdir -p "$OUT"

cd "$DUCK"
echo "=== model=$MODEL_ID served=$SERVED_NAME TP=$TP cases=$MAX_CASES ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- serve vLLM (skip if OPENAI_BASE_URL already set) ----
VLLM_PID=""
if [ -z "${OPENAI_BASE_URL:-}" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --tensor-parallel-size $TP --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$OUT/vllm_${SERVED_NAME}.log" 2>&1 &
  VLLM_PID=$!
  trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
  echo "Waiting for vLLM ($MODEL_ID; first run downloads weights)..."
  for i in $(seq 1 360); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -40 "$OUT/vllm_${SERVED_NAME}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# ---- eval over base set, save traces, compute per-run selection metrics ----
tmp_cfg="$OUT/base_seed0.yaml"
$DUCK_PY -c "from ducx_noise import NoiseConfig; c=NoiseConfig.load('$CONFIG'); c.seed=0; c.label='base'; c.save('$tmp_cfg')"
log_prefix="toolprobe_${SERVED_NAME}_base"
CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
  --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
  --data-file data/chestagentbench/metadata.jsonl --device cuda \
  --log-prefix "$log_prefix" --max-cases "$MAX_CASES" --llm-parse \
  --noise-config "$tmp_cfg"

run_log=$(ls -t "$DUCK/logs/$log_prefix/${log_prefix}_"*.json | head -1)
manifest=$(ls -t "$DUCK/logs/$log_prefix/noise_manifest_"*.json 2>/dev/null | head -1 || true)
$DUCK_PY -m ducx_noise.metrics row --run-log "$run_log" \
  --manifest "${manifest:-}" --label "${SERVED_NAME}" --seed 0 --out "$OUT/results.csv"
echo
echo "=== done. trace log: $run_log ==="
cat "$OUT/results.csv"
