#!/bin/bash
# Confirmatory paired distractor sweep: run several noise configs through ONE
# vLLM server on a dedicated GPU pair, all on the SAME first-N questions of the
# current data file (dataset.select(range(N)) -> identical question set across
# configs -> fully paired). Isolated log prefix so it never mixes with old runs.
#
# Launch two of these on one 4-GPU node for 2 parallel streams:
#   CONFIGS="base distractor_10 distractor_50" VLLM_GPU=0 TOOL_GPU=1 PORT=8000 bash ...
#   CONFIGS="distractor_5 distractor_20"        VLLM_GPU=2 TOOL_GPU=3 PORT=8001 bash ...
set -eo pipefail

export DS=/datasets/omni_pretraining
export DUCK=${DUCK:-/project/6101776/xzhan576/DUCK}
export HF_HOME=${HF_HOME:-$DS/.hf_home}
# The login profile points HF_DATASETS_CACHE at ~/scratch, which is over quota
# (Errno 122). Redirect the datasets cache to /datasets (9TB free) so
# load_dataset() can write its lock/cache. Tool *weights* still load from
# MEDRAX_WEIGHTS via the per-command HF_HOME override below.
export HF_DATASETS_CACHE=$DS/hf_datasets_cache
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$MEDRAX_WEIGHTS" "$MEDRAX_TEMP" "$DUCK/logs"

VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct

CONFIGS=${CONFIGS:?set CONFIGS}
SEED=${SEED:-0}
MAX_CASES=${MAX_CASES:-500}
PORT=${PORT:-8000}
VLLM_GPU=${VLLM_GPU:-0}
TOOL_GPU=${TOOL_GPU:-1}
TAG=${TAG:-sweep500}
RESULTS_DIR=$DUCK/logs/${TAG}
mkdir -p "$RESULTS_DIR"

cd "$DUCK"

# ---- serve vLLM (model under test) on the assigned GPU/port ----
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
  --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$RESULTS_DIR/vllm_p${PORT}.log" 2>&1 &
VLLM_PID=$!
trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
echo "[$TAG p$PORT] waiting for vLLM..."
for i in $(seq 1 240); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "[$TAG p$PORT] vLLM ready"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -30 "$RESULTS_DIR/vllm_p${PORT}.log"; exit 1; }
  sleep 5
done
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY=EMPTY

# ---- run each config through this server on the SAME N questions ----
for cfg in $CONFIGS; do
  label=$cfg
  echo "=== [$TAG p$PORT] config=$label seed=$SEED N=$MAX_CASES ==="
  tmp_cfg="$RESULTS_DIR/${label}_seed${SEED}.yaml"
  $DUCK_PY -c "from ducx_noise import NoiseConfig; c=NoiseConfig.load('ducx_noise/configs/${label}.yaml'); c.seed=$SEED; c.label='$label'; c.save('$tmp_cfg')"
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file data/chestagentbench/metadata.jsonl --device cuda \
    --log-prefix "${TAG}_${label}" --max-cases "$MAX_CASES" --llm-parse \
    --noise-config "$tmp_cfg"
  echo "=== [$TAG p$PORT] DONE $label ==="
done
echo "[$TAG p$PORT] ALL DONE"
