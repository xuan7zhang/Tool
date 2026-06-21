#!/bin/bash
# One-shot smoke test for the reasoning-entropy task (Task-0 verification):
# starts a vLLM Qwen3-VL-8B-Instruct server with the hermes tool parser, fires
# one request asking for logprobs, and checks that per-token logprobs come back
# together with a final answer + tool_calls -- both via the raw OpenAI client
# and via langchain ChatOpenAI (the agent's path, which the capture hook uses).
#
# Run on the 4xL40 node (server uses 1 GPU; 8B is small). From the DUCK repo root:
#   bash ducx_entropy/smoke_test.sh
set -eo pipefail

export DS=/datasets/omni_pretraining
export HF_HOME=$DS/.hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export DUCK=${DUCK:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p "$HF_HOME" "$DUCK/logs"

VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
PORT=${PORT:-8000}
GPU=${GPU:-0}
TP=${TP:-1}

echo "=== GPUs ==="; nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Instruct => no reasoning parser. hermes tool parser + auto tool choice for
# tool_calls. Compute nodes lack nvcc -> disable flashinfer sampler JIT.
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== launching vLLM ($MODEL_ID) on GPU $GPU (TP=$TP) ==="
CUDA_VISIBLE_DEVICES=$GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --port $PORT \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$DUCK/logs/vllm_smoke.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID  log: $DUCK/logs/vllm_smoke.log"
cleanup() { echo "=== stopping vLLM ($VLLM_PID) ==="; kill $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

echo "=== waiting for /v1/models (first run downloads ~16GB weights) ==="
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready after ~$((i*5))s"; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM died. Tail:"; tail -40 "$DUCK/logs/vllm_smoke.log"; exit 1
  fi
  sleep 5
done

cd "$DUCK"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export SMOKE_MODEL=$SERVED_NAME
echo "=== running smoke_client.py ==="
$DUCK_PY ducx_entropy/smoke_client.py
echo "=== smoke done; raw response at /tmp/ducx_smoke_response.json ==="
