#!/bin/bash
# Large-sample task_acc check: is "distractor_5 > base" real or seed noise?
# Runs base and distractor_5 over the same N questions IN PARALLEL on a 4-GPU
# node: stream A = base on GPU0(vLLM)+GPU1(tools) port 8000, stream B =
# distractor_5 on GPU2(vLLM)+GPU3(tools) port 8001. No logprobs (this only needs
# task_accuracy); the run trace is still saved for self-reflection analysis.
#
# Usage (inside a 4-GPU allocation):
#   bash ducx_noise/run_acc500.sh                 # 500 questions, 1 seed (~4h)
#   SEEDS="0 1" bash ducx_noise/run_acc500.sh      # 2 seeds (~7.5h, serial/stream)
#   MAX_CASES=300 bash ducx_noise/run_acc500.sh    # fewer questions
set -uo pipefail

export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
DUCK_PY=/datasets/omni_pretraining/.venvs/DUCK/bin/python
cd "$DUCK"
unset OPENAI_BASE_URL  # force each stream to serve its own vLLM

MAX_CASES=${MAX_CASES:-500}
SEEDS=${SEEDS:-0}
OUT=${OUT:-$DUCK/logs/acc500}
mkdir -p "$OUT"

echo "=== launching 2 parallel streams (MAX_CASES=$MAX_CASES SEEDS='$SEEDS') ==="

CONFIGS="ducx_noise/configs/base.yaml" SEEDS="$SEEDS" MAX_CASES=$MAX_CASES \
  VLLM_GPU=0 TOOL_GPU=1 PORT=8000 RESULTS_DIR="$OUT/base" \
  bash ducx_noise/run_noise_experiment.sh > "$OUT/base.out" 2>&1 &
PIDA=$!

CONFIGS="ducx_noise/configs/distractor_5.yaml" SEEDS="$SEEDS" MAX_CASES=$MAX_CASES \
  VLLM_GPU=2 TOOL_GPU=3 PORT=8001 RESULTS_DIR="$OUT/d5" \
  bash ducx_noise/run_noise_experiment.sh > "$OUT/d5.out" 2>&1 &
PIDB=$!

echo "Stream A (base)        PID=$PIDA  log: $OUT/base.out"
echo "Stream B (distractor_5) PID=$PIDB  log: $OUT/d5.out"
echo "Tail progress with: tail -f $OUT/base.out  /  $OUT/d5.out"

rc=0
wait $PIDA || { echo "stream A failed"; rc=1; }
wait $PIDB || { echo "stream B failed"; rc=1; }

echo
echo "=== both streams done; base vs distractor_5 task_acc ==="
$DUCK_PY -m ducx_noise.metrics aggregate \
  --in "$OUT/base/results.csv" "$OUT/d5/results.csv" \
  --summary-out "$OUT/summary.md" --summary-csv "$OUT/summary.csv"
echo
cat "$OUT/summary.md"
echo
echo "Per-run rows: $OUT/base/results.csv , $OUT/d5/results.csv"
exit $rc
