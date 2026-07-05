#!/bin/bash
# Extended toolbias sweeps on the tool-necessary probe task (reuses the probe
# dataset + the clean/no_tool baselines from run_toolbias_experiment.sh).
#
#   MODE=count     -> distractor count dose-response (random position, shuffle),
#                     for both similarity tiers.
#   MODE=position  -> Task-5 position ablation: fixed count/similarity, tool_order
#                     controlled, sweep head|tail|random|index=k.
#
# Usage (inside the GPU alloc, one stream per MODE):
#   MODE=count    VLLM_GPU=0 TOOL_GPU=1 PORT=8000 bash ducx_noise/run_toolbias_sweep.sh
#   MODE=position VLLM_GPU=2 TOOL_GPU=3 PORT=8001 bash ducx_noise/run_toolbias_sweep.sh
set -eo pipefail

export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}
VLLM_GPU=${VLLM_GPU:-0}
TOOL_GPU=${TOOL_GPU:-1}
MODE=${MODE:-count}
SEED=${SEED:-0}
MAX_CASES=${MAX_CASES:-50}
CAPTURE_LOGPROBS=${CAPTURE_LOGPROBS:-20}
MODEL_DTYPE=${MODEL_DTYPE:-fp16}
DECODING=${DECODING:-greedy}
TOOLS=${TOOLS:-"ImageVisualizerTool,DicomProcessorTool,ChestXRayClassifierTool"}
PROBE_FILE=${PROBE_FILE:-$DUCK/logs/toolbias/probe_dataset.jsonl}
RESULTS_DIR=${RESULTS_DIR:-$DUCK/logs/toolbias_sweep}
mkdir -p "$RESULTS_DIR"
VLLM_DTYPE=float16; [ "$MODEL_DTYPE" = bf16 ] && VLLM_DTYPE=bfloat16
cd "$DUCK"

# ---- serve vLLM ------------------------------------------------------------
if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype "$VLLM_DTYPE" --seed "$SEED" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_sweep_${MODE}_p${PORT}.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ($MODE)"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED"; tail -30 "$DUCK/logs/vllm_sweep_${MODE}_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

if [ ! -s "$PROBE_FILE" ]; then
  echo "Building probe dataset -> $PROBE_FILE"
  CUDA_VISIBLE_DEVICES=$TOOL_GPU $DUCK_PY -m ducx_noise.probe_task \
    --source data/chestagentbench/metadata.jsonl --out-file "$PROBE_FILE" \
    --n "$MAX_CASES" --device cuda --seed "$SEED"
fi

# gen_cfg <path> <count> <position> <index> <similarity> <order> <tag>
gen_cfg() {
  $DUCK_PY - "$@" <<'PY'
import sys
from ducx_noise.config import NoiseConfig, DistractorConfig
p,cnt,pos,idx,sim,order,tag = sys.argv[1:8]
c = NoiseConfig(seed=0, tool_order=order, label=tag,
    distractor=DistractorConfig(enabled=True, count=int(cnt), position=pos,
        index=int(idx), similarity=sim, llm_generate=(sim=="aligned")))
c.save(p)
PY
}

run_cfg() {  # run_cfg <prefix> <condition> [extra launcher flags...]
  local prefix="$1" cond="$2"; shift 2
  echo "=== $prefix ($cond) ==="
  # capture-logprobs only when CAPTURE_LOGPROBS>0 (0 keeps memory low for big-N
  # accuracy sweeps; the likelihood angle is covered by the small-N runs).
  local lp_flag=()
  [ "${CAPTURE_LOGPROBS:-0}" -gt 0 ] && lp_flag=(--capture-logprobs "$CAPTURE_LOGPROBS")
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$PROBE_FILE" --device cuda \
    --log-prefix "$prefix" --max-cases "$MAX_CASES" --llm-parse \
    --decoding "$DECODING" --seed "$SEED" --model-dtype "$MODEL_DTYPE" \
    "${lp_flag[@]}" \
    --path tool_useful_distractor --condition "$cond" "$@"
}

# tag suffix so 500-case sweeps do not collide with the 50-case dirs
SUF=${SUF:-n${MAX_CASES}}

if [ "$MODE" = count ]; then
  for sim in obvious aligned; do
    for cnt in 1 2 3 5 10 20; do
      cfg="$RESULTS_DIR/count_c${cnt}_${sim}_${SUF}.yaml"
      gen_cfg "$cfg" "$cnt" random 0 "$sim" shuffle "count_c${cnt}_${sim}"
      run_cfg "sw_count_c${cnt}_${sim}_${SUF}" tool_useful_distractor \
        --tools "$TOOLS" --noise-config "$cfg"
    done
  done
elif [ "$MODE" = position ]; then
  # baselines first (reference lines): no_tool + clean tool_useful.
  # SKIP_BASE=1 when they are already done (avoid recomputing 2x500).
  if [ "${SKIP_BASE:-0}" != "1" ]; then
    run_cfg "sw_base_no_tool_${SUF}" no_tool --disable-tools
    run_cfg "sw_base_clean_${SUF}" tool_useful --tools "$TOOLS"
  fi
  for spec in head:0 tail:0 random:0 index:0 index:3; do
    pos="${spec%%:*}"; idx="${spec##*:}"
    tag="pos_${pos}${idx}"
    cfg="$RESULTS_DIR/${tag}_${SUF}.yaml"
    gen_cfg "$cfg" 5 "$pos" "$idx" aligned controlled "$tag"
    run_cfg "sw_${tag}_${SUF}" tool_useful_distractor --tools "$TOOLS" --noise-config "$cfg"
  done
else
  echo "Unknown MODE=$MODE (want count|position)"; exit 1
fi
echo "=== SWEEP $MODE DONE ==="
