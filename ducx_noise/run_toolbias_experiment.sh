#!/bin/bash
# Stage 4 runner: the two experiment paths x their conditions, each producing
# Stage-5 schema logs (with per-token logprobs) for the Stage-6 analysis.
#
#   path=tool_useless            -> ChestAgentBench (tools do not help the query)
#       conditions: no_tool, tool_only, tool_useless
#   path=tool_useful_distractor  -> probe_task MCQ (answer determined by the tool)
#       conditions: no_tool, tool_useful, tool_useful_distractor(sweep)
#
# Every condition is run with-tool AND no-tool so Stage 6 can do the likelihood
# comparison. Greedy + fp16 + fixed seed come from Stage 1.
#
# Usage (inside a GPU allocation, e.g. srun --jobid=<ID> --overlap --ntasks=1 ...):
#   MAX_CASES=20 PATHS="tool_useless tool_useful_distractor" \
#   bash ducx_noise/run_toolbias_experiment.sh
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

# --- experiment knobs -------------------------------------------------------
PATHS=${PATHS:-"tool_useless tool_useful_distractor"}
SEED=${SEED:-0}
MAX_CASES=${MAX_CASES:-20}
CAPTURE_LOGPROBS=${CAPTURE_LOGPROBS:-20}    # top-K logprobs for likelihood
MODEL_DTYPE=${MODEL_DTYPE:-fp16}
DECODING=${DECODING:-greedy}
# distractor sweep for the tool_useful_distractor path:
COUNTS=${COUNTS:-"2 5"}
POSITIONS=${POSITIONS:-"random"}
SIMILARITIES=${SIMILARITIES:-"obvious aligned"}
DATA_USELESS=${DATA_USELESS:-data/chestagentbench/metadata.jsonl}
PROBE_FILE=${PROBE_FILE:-$DUCK/logs/toolbias/probe_dataset.jsonl}
# Tool set exposed to the agent. Default to the light, proven set (classifier is
# the tool-necessary one for the probe task); empty string -> launcher default.
TOOLS=${TOOLS:-"ImageVisualizerTool,DicomProcessorTool,ChestXRayClassifierTool"}

case "$MODEL_DTYPE" in
  fp16) VLLM_DTYPE=float16 ;;
  bf16) VLLM_DTYPE=bfloat16 ;;
  *) echo "Unknown MODEL_DTYPE=$MODEL_DTYPE"; exit 1 ;;
esac

RESULTS_DIR=${RESULTS_DIR:-$DUCK/logs/toolbias}
mkdir -p "$RESULTS_DIR"
cd "$DUCK"

# ---- 1. serve vLLM (skip if OPENAI_BASE_URL already set) --------------------
VLLM_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --dtype "$VLLM_DTYPE" --seed "$SEED" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_toolbias_${SLURM_JOB_ID:-local}_p${PORT}.log" 2>&1 &
  VLLM_PID=$!
  trap '[ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true' EXIT
  echo "Waiting for vLLM..."
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -30 "$DUCK/logs/vllm_toolbias_"*.log; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# helper: one launch_over_chexbench run with the shared reproducibility flags.
#   run_eval <log_prefix> <path> <condition> <data_file> [extra flags...]
run_eval() {
  local prefix="$1" path="$2" cond="$3" data="$4"; shift 4
  echo "=== path=$path condition=$cond prefix=$prefix ==="
  local tools_flag=()
  # no_tool condition uses --disable-tools (passed via "$@"); otherwise restrict
  # to the configured TOOLS set unless the caller already passed --disable-tools.
  if [[ -n "$TOOLS" && "$*" != *"--disable-tools"* ]]; then
    tools_flag=(--tools "$TOOLS")
  fi
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$data" --device cuda \
    --log-prefix "$prefix" --max-cases "$MAX_CASES" --llm-parse \
    --decoding "$DECODING" --seed "$SEED" --model-dtype "$MODEL_DTYPE" \
    --capture-logprobs "$CAPTURE_LOGPROBS" \
    "${tools_flag[@]}" \
    --path "$path" --condition "$cond" "$@"
}

# ---- 2. path: tool_useless (ChestAgentBench) -------------------------------
if echo "$PATHS" | grep -qw tool_useless; then
  # no-tool baseline
  run_eval "tb_useless_no_tool_s${SEED}" tool_useless no_tool "$DATA_USELESS" --disable-tools
  # with-tool (real tools present, no noise)
  run_eval "tb_useless_tool_only_s${SEED}" tool_useless tool_only "$DATA_USELESS"
  # tool_useless (same env, semantic label: tools present but unhelpful for the query)
  run_eval "tb_useless_useless_s${SEED}" tool_useless tool_useless "$DATA_USELESS"
fi

# ---- 3. path: tool_useful_distractor (probe MCQ) ---------------------------
if echo "$PATHS" | grep -qw tool_useful_distractor; then
  # Build the tool-necessary probe dataset (answer determined by the real
  # classifier; guessing without the tool is ~chance).
  if [ ! -s "$PROBE_FILE" ]; then
    echo "=== building probe dataset -> $PROBE_FILE ==="
    CUDA_VISIBLE_DEVICES=$TOOL_GPU $DUCK_PY -m ducx_noise.probe_task \
      --source "$DATA_USELESS" --out-file "$PROBE_FILE" --n "$MAX_CASES" \
      --device cuda --seed "$SEED"
  fi
  # no-tool baseline on the probe task (expected ~chance)
  run_eval "tb_useful_no_tool_s${SEED}" tool_useful_distractor no_tool "$PROBE_FILE" --disable-tools
  # tool_useful (real tools, no distractor)
  run_eval "tb_useful_clean_s${SEED}" tool_useful_distractor tool_useful "$PROBE_FILE"
  # tool_useful_distractor: sweep count x position x similarity
  for sim in $SIMILARITIES; do
    for pos in $POSITIONS; do
      for cnt in $COUNTS; do
        tag="c${cnt}_${pos}_${sim}"
        cfg="$RESULTS_DIR/distractor_${tag}_s${SEED}.yaml"
        $DUCK_PY - "$cfg" "$cnt" "$pos" "$sim" "$SEED" "$tag" <<'PY'
import sys
from ducx_noise.config import NoiseConfig, DistractorConfig
cfg_path, cnt, pos, sim, seed, tag = sys.argv[1:7]
c = NoiseConfig(
    seed=int(seed), tool_order="shuffle", label=tag,
    distractor=DistractorConfig(enabled=True, count=int(cnt), position=pos,
                                similarity=sim, llm_generate=(sim == "aligned")),
)
c.save(cfg_path)
PY
        run_eval "tb_useful_${tag}_s${SEED}" tool_useful_distractor \
          tool_useful_distractor "$PROBE_FILE" --noise-config "$cfg"
      done
    done
  done
fi

echo
echo "=== toolbias runs done. logs under $DUCK/logs/tb_* ==="
echo "Next: python -m analysis.toolbias_analysis --logs-glob '$DUCK/logs/tb_*/tb_*_*.json' --out-dir analysis/toolbias"
