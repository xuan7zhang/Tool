#!/bin/bash
# One-command reasoning-entropy experiment (opt-in; reuses the noise pipeline):
#   serve vLLM -> re-run agent eval per (noise config x seed) WITH --capture-logprobs
#   -> ducx_noise.metrics row (selection entropy) -> ducx_entropy.cli extract
#   (reasoning likelihood/entropy, granularities A/B/C) -> ducx_entropy.analyze
#   (reasoning vs selection comparison table).
#
# Usage (inside a GPU allocation):
#   CONFIGS="ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml" \
#   SEEDS="0 1 2" MAX_CASES=20 TOPK=20 bash ducx_entropy/run_entropy_experiment.sh
#
# Stays on Qwen3-VL-8B-Instruct; reasoning segment = the final-answer prose.
set -eo pipefail

export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
mkdir -p "$HF_HOME" "$MEDRAX_WEIGHTS" "$MEDRAX_TEMP" "$DUCK/logs"

VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}
VLLM_GPU=${VLLM_GPU:-0}
TOOL_GPU=${TOOL_GPU:-1}

CONFIGS=${CONFIGS:-"ducx_noise/configs/base.yaml ducx_noise/configs/distractor_5.yaml"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_CASES=${MAX_CASES:-20}
TOPK=${TOPK:-20}
RESULTS_DIR=${RESULTS_DIR:-$DUCK/logs/entropy_experiment}
RESULTS_CSV=$RESULTS_DIR/selection_results.csv
ENTROPY_DIR=$RESULTS_DIR/entropy
mkdir -p "$RESULTS_DIR" "$ENTROPY_DIR"
rm -f "$RESULTS_CSV"

cd "$DUCK"

# ---- 1. serve vLLM (skip if OPENAI_BASE_URL already set) ----
VLLM_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_entropy_${SLURM_JOB_ID:-local}_p${PORT}.log" 2>&1 &
  VLLM_PID=$!
  trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
  echo "Waiting for vLLM..."
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -30 "$DUCK/logs/vllm_entropy_"*.log; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# ---- 2. per (config, seed): eval with logprobs -> selection row -> entropy extract ----
for cfg in $CONFIGS; do
  label=$(basename "$cfg" | sed 's/\.[^.]*$//')
  for seed in $SEEDS; do
    echo "=== config=$label seed=$seed (capture-logprobs=$TOPK) ==="
    tmp_cfg="$RESULTS_DIR/${label}_seed${seed}.yaml"
    $DUCK_PY -c "from ducx_noise import NoiseConfig; c=NoiseConfig.load('$cfg'); c.seed=$seed; c.label='$label'; c.save('$tmp_cfg')"
    log_prefix="entropy_${label}_seed${seed}"
    CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
      --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
      --data-file data/chestagentbench/metadata.jsonl --device cuda \
      --log-prefix "$log_prefix" --max-cases "$MAX_CASES" --llm-parse \
      --noise-config "$tmp_cfg" --capture-logprobs "$TOPK"
    run_log=$(ls -t "$DUCK/logs/$log_prefix/${log_prefix}_"*.json | head -1)
    saved_manifest=$(ls -t "$DUCK/logs/$log_prefix/noise_manifest_"*.json 2>/dev/null | head -1 || true)
    $DUCK_PY -m ducx_noise.metrics row --run-log "$run_log" \
      --manifest "${saved_manifest:-}" --label "$label" --seed "$seed" --out "$RESULTS_CSV"
    $DUCK_PY -m ducx_entropy.cli extract --run-log "$run_log" \
      --label "$label" --seed "$seed" --out-dir "$ENTROPY_DIR"
  done
done

# ---- 3. compare reasoning entropy vs selection entropy ----
echo
echo "=== reasoning vs selection (mean over seeds) ==="
$DUCK_PY -m ducx_entropy.analyze --perturn "$ENTROPY_DIR"/*_perturn.jsonl \
  --selection-csv "$RESULTS_CSV" \
  --out-md "$RESULTS_DIR/reasoning_vs_selection.md" \
  --out-csv "$RESULTS_DIR/reasoning_vs_selection.csv"
echo
echo "Selection rows : $RESULTS_CSV"
echo "Entropy JSONL  : $ENTROPY_DIR/"
echo "Comparison     : $RESULTS_DIR/reasoning_vs_selection.md"
