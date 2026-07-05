#!/bin/bash
# Gating experiment: on a MIXED set (text-answerable + tool-necessary), does a
# per-query tool GATE beat both always-tools and never-tools? Tool value flips
# sign with tool-necessity, so a gate that matches exposure to necessity should
# win. Run two conditions on the same 400 questions; 'gated' is derived per-
# question (text->never, tool->always) so no third run is needed.
set -eo pipefail
export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
VLLM_PY=$DS/.venvs/vllm/bin/python; DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b; MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}; VLLM_GPU=${VLLM_GPU:-0}; TOOL_GPU=${TOOL_GPU:-1}; SEED=${SEED:-0}
PROBE=${PROBE:-$DUCK/logs/toolbias/gating_mixed.jsonl}
TOOLS=${TOOLS:-ChestXRayClassifierTool,ChestXRaySegmentationTool}
cd "$DUCK"

if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype float16 --seed "$SEED" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_gating_p${PORT}.log" 2>&1 &
  VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VP 2>/dev/null || { echo "vLLM DIED"; tail -20 "$DUCK/logs/vllm_gating_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

run() {  # run <prefix> <condition> [extra...]
  local prefix="$1" cond="$2"; shift 2
  echo "=== $prefix ($cond) ==="
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$PROBE" --device cuda --log-prefix "$prefix" \
    --llm-parse --decoding greedy --seed "$SEED" --model-dtype fp16 \
    --path gating --condition "$cond" "$@"
}

run gate_never   never_tool  --disable-tools
run gate_always  always_tool --tools "$TOOLS"
echo "=== GATING DONE ==="
