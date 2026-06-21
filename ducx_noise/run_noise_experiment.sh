#!/bin/bash
# One-command noisy-environment experiment:
#   serve vLLM -> run agent eval per (noise config x seed) -> compute & aggregate
#   noise selection metrics -> print selection accuracy + noise-tool mis-selection.
#
# Usage (inside a GPU allocation, e.g. srun --jobid=<ID> --overlap --ntasks=1 ...):
#   CONFIGS="ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml" \
#   SEEDS="0 1 2" MAX_CASES=20 bash ducx_noise/run_noise_experiment.sh
#
# Defaults reproduce a quick base-vs-distractor_5 smoke comparison.
set -eo pipefail

export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=/project/aip-xli135/xzhan576/DUCK
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
mkdir -p "$HF_HOME" "$MEDRAX_WEIGHTS" "$MEDRAX_TEMP" "$DUCK/logs"

VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}
VLLM_GPU=${VLLM_GPU:-0}   # GPU index for the vLLM server
TOOL_GPU=${TOOL_GPU:-1}   # GPU index for the agent tools
# Lets two streams share one 4-GPU node: stream A (GPU0/1,port8000) + B (GPU2/3,port8001).

CONFIGS=${CONFIGS:-"ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_CASES=${MAX_CASES:-20}
RESULTS_DIR=${RESULTS_DIR:-$DUCK/logs/noise_experiment}
RESULTS_CSV=$RESULTS_DIR/results.csv
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_CSV"

cd "$DUCK"

# ---- 1. serve vLLM on GPU 0 (skip if OPENAI_BASE_URL already points at a server) ----
VLLM_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_noise_${SLURM_JOB_ID:-local}_p${PORT}.log" 2>&1 &
  VLLM_PID=$!
  trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
  echo "Waiting for vLLM..."
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -30 "$DUCK/logs/vllm_noise_"*.log; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# ---- 2. run eval per (config, seed) on GPU 1, compute metrics row ----
for cfg in $CONFIGS; do
  label=$(basename "$cfg" | sed 's/\.[^.]*$//')
  for seed in $SEEDS; do
    echo "=== config=$label seed=$seed ==="
    # Stamp the seed into a temp config so runs are independent + reproducible.
    tmp_cfg="$RESULTS_DIR/${label}_seed${seed}.yaml"
    $DUCK_PY -c "import sys; from ducx_noise import NoiseConfig; c=NoiseConfig.load('$cfg'); c.seed=$seed; c.label='$label'; c.save('$tmp_cfg')"
    log_prefix="noise_${label}_seed${seed}"
    manifest="$RESULTS_DIR/${log_prefix}_manifest.json"
    CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
      --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
      --data-file data/chestagentbench/metadata.jsonl --device cuda \
      --log-prefix "$log_prefix" --max-cases "$MAX_CASES" --llm-parse \
      --noise-config "$tmp_cfg"
    run_log=$(ls -t "$DUCK/logs/$log_prefix/${log_prefix}_"*.json | head -1)
    saved_manifest=$(ls -t "$DUCK/logs/$log_prefix/noise_manifest_"*.json 2>/dev/null | head -1 || true)
    $DUCK_PY -m ducx_noise.metrics row --run-log "$run_log" \
      --manifest "${saved_manifest:-}" --label "$label" --seed "$seed" --out "$RESULTS_CSV"
  done
done

# ---- 3. aggregate base-vs-noisy summary ----
echo
echo "=== Aggregated summary (mean over seeds) ==="
$DUCK_PY -m ducx_noise.metrics aggregate --in "$RESULTS_CSV" \
  --summary-out "$RESULTS_DIR/summary.md" --summary-csv "$RESULTS_DIR/summary.csv"
echo
echo "Per-run rows: $RESULTS_CSV"
echo "Summary table: $RESULTS_DIR/summary.md"
