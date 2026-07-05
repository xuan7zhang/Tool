#!/bin/bash
#SBATCH --job-name=duck_chexbench
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/project/aip-xli135/xzhan576/DUCK/logs/slurm_%j.out

set -eo pipefail

# ---- paths / caches all on /datasets (10T) to avoid home/project quota ----
export DS=/datasets/omni_pretraining
export HF_HOME=$DS/.hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export DUCK=/project/aip-xli135/xzhan576/DUCK
export MEDRAX_WEIGHTS=$DS/.medrax_weights
export MEDRAX_TEMP=$DS/.medrax_temp
mkdir -p "$HF_HOME" "$MEDRAX_WEIGHTS" "$MEDRAX_TEMP" "$DUCK/logs"

VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=8000
# Reproducibility knobs (Stage 1): fp16 load + deterministic decoding + fixed seed.
MODEL_DTYPE=${MODEL_DTYPE:-fp16}          # fp16 | bf16
DECODING=${DECODING:-greedy}              # greedy | sample
SEED=${SEED:-0}
# Map fp16/bf16 -> vLLM --dtype value.
case "$MODEL_DTYPE" in
  fp16) VLLM_DTYPE=float16 ;;
  bf16) VLLM_DTYPE=bfloat16 ;;
  *) echo "Unknown MODEL_DTYPE=$MODEL_DTYPE"; exit 1 ;;
esac

echo "=== GPUs visible ==="; nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- 1. start vLLM server on GPU 0 ----
# Compute nodes lack nvcc/CUDA toolkit, so disable flashinfer JIT paths (sampler);
# vLLM falls back to prebuilt native kernels that need no runtime nvcc compile.
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== launching vLLM ($MODEL_ID) on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --port $PORT \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --dtype "$VLLM_DTYPE" \
  --seed "$SEED" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$DUCK/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID, log: $DUCK/logs/vllm_${SLURM_JOB_ID}.log"

cleanup() { echo "=== stopping vLLM ($VLLM_PID) ==="; kill $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

# ---- 2. wait until server healthy (up to ~20 min for first-time weight download) ----
echo "=== waiting for vLLM /v1/models ==="
for i in $(seq 1 240); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready after ~$((i*5))s"; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died. Tail of log:"; tail -40 "$DUCK/logs/vllm_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
curl -s "http://localhost:$PORT/v1/models" || { echo "vLLM never became healthy"; exit 1; }
echo

# ---- 3. run ChestAgentBench agent eval on GPU 1 (small reproduction) ----
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
cd "$DUCK"
echo "=== running launch_over_chexbench.py (--max-cases ${MAX_CASES:-3}) ==="
CUDA_VISIBLE_DEVICES=1 HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
  --model "$SERVED_NAME" \
  --model-dir "$MEDRAX_WEIGHTS" \
  --temp-dir "$MEDRAX_TEMP" \
  --data-file data/chestagentbench/metadata.jsonl \
  --device cuda \
  --log-prefix qwen3vl8b-vllm \
  --max-cases "${MAX_CASES:-3}" \
  --decoding "$DECODING" \
  --seed "$SEED" \
  --model-dtype "$MODEL_DTYPE" \
  --llm-parse

echo "=== eval done. logs under $DUCK/logs/qwen3vl8b-vllm/ ==="
ls -la "$DUCK/logs/qwen3vl8b-vllm/" || true
