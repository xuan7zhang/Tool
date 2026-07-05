#!/bin/bash
# Establish the oracle bounds for the multi-tool probe suite: how much accuracy a
# per-query tool-space optimizer could recover. Same backbone/judgment throughout;
# only the TOOL EXPOSURE changes.
#
#   no_tool          -- no tools (validates tool-necessity: should be ~chance)
#   oracle           -- expose ONLY the query's required tool (per-question) [upper bound]
#   all_real         -- expose BOTH real tools (does the irrelevant real tool pollute?)
#   all_real_distract-- both real tools + N distractors                       [lower bound]
#
# Retriever headroom = oracle - all_real_distract (and does pruning beat all_real?).
#
# Usage (inside a GPU alloc):
#   PROBE=logs/toolbias/multitool_probe_n120.jsonl VLLM_GPU=0 TOOL_GPU=1 \
#   bash ducx_noise/run_multitool_bounds.sh
set -eo pipefail

export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
mkdir -p "$HF_HOME" "$MEDRAX_WEIGHTS" "$MEDRAX_TEMP" "$DUCK/logs"
VLLM_PY=$DS/.venvs/vllm/bin/python
DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b; MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}; VLLM_GPU=${VLLM_GPU:-0}; TOOL_GPU=${TOOL_GPU:-1}
SEED=${SEED:-0}; MAX_CASES=${MAX_CASES:-100000}
PROBE=${PROBE:-$DUCK/logs/toolbias/multitool_probe_n120.jsonl}
CLS_TOOL=ChestXRayClassifierTool; SEG_TOOL=ChestXRaySegmentationTool
RESULTS=${RESULTS:-$DUCK/logs/toolbias/multitool}; mkdir -p "$RESULTS"
cd "$DUCK"

# split the probe file by required_tool for the per-question oracle
CLS_FILE="$RESULTS/probe_cls.jsonl"; SEG_FILE="$RESULTS/probe_seg.jsonl"
$DUCK_PY - "$PROBE" "$CLS_FILE" "$SEG_FILE" <<'PY'
import sys, json
src, clsf, segf = sys.argv[1:4]
with open(clsf,"w") as c, open(segf,"w") as s:
    for l in open(src):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        (c if r.get("probe_type")=="CLS" else s).write(l+"\n")
print("split done")
PY

# ---- serve vLLM (fp16 greedy seed) ----
if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype float16 --seed "$SEED" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_multitool_p${PORT}.log" 2>&1 &
  VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VP 2>/dev/null || { echo "vLLM DIED"; tail -30 "$DUCK/logs/vllm_multitool_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

run() {  # run <prefix> <condition> <data> [extra flags...]
  local prefix="$1" cond="$2" data="$3"; shift 3
  echo "=== $prefix ($cond) ==="
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$data" --device cuda --log-prefix "$prefix" --max-cases "$MAX_CASES" \
    --llm-parse --decoding greedy --seed "$SEED" --model-dtype fp16 \
    --path multitool --condition "$cond" "$@"
}

# 1) no_tool (tool-necessity check)
run mt_no_tool no_tool "$PROBE" --disable-tools
# 2) oracle: each subset sees ONLY its required tool
run mt_oracle_cls oracle "$CLS_FILE" --tools "$CLS_TOOL"
run mt_oracle_seg oracle "$SEG_FILE" --tools "$SEG_TOOL"
# 3) all_real: both real tools exposed for every question
run mt_all_real all_real "$PROBE" --tools "$CLS_TOOL,$SEG_TOOL"
# 4) all_real + distractors (lower bound)
DCFG="$RESULTS/distract5.yaml"
$DUCK_PY - "$DCFG" <<'PY'
import sys
from ducx_noise.config import NoiseConfig, DistractorConfig
NoiseConfig(seed=0, tool_order="shuffle", label="mt_distract5",
    distractor=DistractorConfig(enabled=True, count=5, position="random",
        similarity="obvious")).save(sys.argv[1])
PY
run mt_all_distract all_real_distract "$PROBE" --tools "$CLS_TOOL,$SEG_TOOL" --noise-config "$DCFG"

echo "=== MULTITOOL BOUNDS DONE ==="
