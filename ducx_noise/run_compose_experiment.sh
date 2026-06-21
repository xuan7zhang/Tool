#!/bin/bash
# One-command compositional (K-operator) experiment:
#   mine pair -> write condition configs -> serve vLLM -> precompute p_b embeddings
#   -> run agent eval for each condition (p_b / p_c / p_chain / both_macro)
#   -> summarize the three p's + criteria table.
#
# Run inside a GPU allocation, e.g.
#   MAX_CASES=20 srun --jobid=<ID> --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
#       bash ducx_noise/run_compose_experiment.sh
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
MAX_CASES=${MAX_CASES:-20}
DATA=data/chestagentbench/metadata.jsonl

OUT=$DUCK/logs/compose_experiment
mkdir -p "$OUT"
cd "$DUCK"

# ---- optional: tool-necessary probe task (PROBE=1) --------------------------
# ChestAgentBench MCQs are answerable from clinical text, so the tool is not
# necessary and p_b is inflated. PROBE=1 builds a task whose answer = the real
# classifier's argmax over its own top-6 pathologies -> the tool is required.
if [ "${PROBE:-0}" = "1" ]; then
  echo "=== building tool-necessary probe dataset (n=$MAX_CASES) ==="
  PROBE_DATA="$OUT/probe_data.jsonl"
  CUDA_VISIBLE_DEVICES=1 HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY -m ducx_noise.probe_task \
    --source "$DATA" --out-file "$PROBE_DATA" --n "$MAX_CASES" --device cuda \
    --figures-root . --seed 0
  DATA="$PROBE_DATA"
fi

# ---- 0. mine pairs + write condition configs (CPU, quick) ----
$DUCK_PY -m ducx_noise.mine_pairs --out "$OUT/mined_pairs.csv"
$DUCK_PY -m ducx_noise.eval_compose configs --out-dir "$OUT/configs" --device cuda

# ---- 1. serve vLLM on GPU 0 ----
VLLM_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=0 $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_compose_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PID=$!
  trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
  echo "Waiting for vLLM..."
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -30 "$DUCK/logs/vllm_compose_"*.log; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# ---- 2. precompute p_b data (inject t_a's real embedding into each question) ----
PB_DATA="$OUT/pb_data.jsonl"
CUDA_VISIBLE_DEVICES=1 $DUCK_PY -m ducx_noise.eval_compose precompute-pb \
  --data-file "$DATA" --out-file "$PB_DATA" --max-cases "$MAX_CASES" --device cuda --figures-root .

# ---- 3. run each condition on GPU 1 (--tools __none__ avoids loading the 8 base tools) ----
run_condition () {  # $1=kind  $2=data_file
  local kind=$1 data=$2
  local prefix="compose_${kind}"
  echo "=== condition $kind ==="
  CUDA_VISIBLE_DEVICES=1 HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$data" --device cuda --tools __none__ \
    --log-prefix "$prefix" --max-cases "$MAX_CASES" --llm-parse \
    --noise-config "$OUT/configs/${kind}.yaml"
}
run_condition p_b "$PB_DATA"
run_condition p_c "$DATA"
run_condition p_chain "$DATA"
run_condition both_macro "$DATA"

# ---- 4. summarize ----
RUNS_JSON="$OUT/runs.json"
$DUCK_PY - "$OUT" > "$RUNS_JSON" <<'PY'
import sys, glob, os, json
out = sys.argv[1]
runs = {}
for kind in ["p_b", "p_c", "p_chain", "both_macro"]:
    d = os.path.join(out, "..", f"compose_{kind}")
    logs = sorted(glob.glob(os.path.join(os.path.dirname(out), f"compose_{kind}", f"compose_{kind}_*.json")))
    mans = sorted(glob.glob(os.path.join(os.path.dirname(out), f"compose_{kind}", "noise_manifest_*.json")))
    if logs:
        runs[kind] = {"run_log": logs[-1], "manifest": mans[-1] if mans else None}
print(json.dumps(runs))
PY
$DUCK_PY -m ducx_noise.eval_compose summarize --runs "$RUNS_JSON" --out "$OUT/summary.md"
echo
echo "Summary -> $OUT/summary.md"
