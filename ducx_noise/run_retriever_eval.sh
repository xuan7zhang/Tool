#!/bin/bash
# Serve a vLLM and run the LLM-router retriever on the multi-tool probe suite.
# Tests: does the model itself identify the required tool (router accuracy)?
# Lexical retrieval is done offline (no server needed) and is ~perfect here.
set -eo pipefail
export DS=/datasets/omni_pretraining
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
VLLM_PY=$DS/.venvs/vllm/bin/python; DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b; MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}; VLLM_GPU=${VLLM_GPU:-0}
DATASET=${DATASET:-$DUCK/logs/toolbias/multitool_probe.jsonl}
REGISTRY=${REGISTRY:-$DUCK/logs/toolbias/registry.json}
TOPK=${TOPK:-1}; OUT=${OUT:-$DUCK/logs/toolbias/retriever_llm}
cd "$DUCK"

if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype float16 --seed 0 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_retriever_p${PORT}.log" 2>&1 &
  VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VP 2>/dev/null || { echo "vLLM DIED"; tail -20 "$DUCK/logs/vllm_retriever_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

echo "=== LLM-router on $(basename "$DATASET") ==="
$DUCK_PY -m ducx_noise.retriever eval --dataset "$DATASET" --registry "$REGISTRY" \
  --method llm --topk "$TOPK" --model "$SERVED_NAME" --out-dir "$OUT"
echo "=== RETRIEVER EVAL DONE ==="
