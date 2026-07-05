#!/bin/bash
# Nail the dissociation: the model KNOWS which tool it needs (LLM-router top1=1.0)
# but cannot SUPPRESS irrelevant tools in context. Paired, same-question conditions:
#
#   no_tool     -- floor
#   oracle      -- expose ONLY the required tool (per question)      [external pruning]
#   all_real    -- expose both real tools (no intervention)          [degraded]
#   self_route  -- expose both real tools + instruct the model to pick ONE tool and
#                  ignore the others, THEN answer                    [self-rescue test]
#
# Prediction: self_route ~ all_real << oracle  =>  self-routing does NOT rescue;
# only enforced external pruning does. Greedy => deterministic => paired McNemar.
set -eo pipefail
export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
VLLM_PY=$DS/.venvs/vllm/bin/python; DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b; MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}; VLLM_GPU=${VLLM_GPU:-0}; TOOL_GPU=${TOOL_GPU:-1}
SEED=${SEED:-0}
PROBE=${PROBE:-$DUCK/logs/toolbias/multitool_probe_n300.jsonl}
CLS_TOOL=ChestXRayClassifierTool; SEG_TOOL=ChestXRaySegmentationTool
RESULTS=${RESULTS:-$DUCK/logs/toolbias/dissociation}; mkdir -p "$RESULTS"
cd "$DUCK"

# split for the per-question oracle + build the self_route variant
CLS_FILE="$RESULTS/probe_cls.jsonl"; SEG_FILE="$RESULTS/probe_seg.jsonl"
ROUTE_FILE="$RESULTS/probe_selfroute.jsonl"
$DUCK_PY - "$PROBE" "$CLS_FILE" "$SEG_FILE" "$ROUTE_FILE" <<'PY'
import sys, json
src, clsf, segf, routef = sys.argv[1:5]
INSTR = ("IMPORTANT: Several tools are available, but only ONE is relevant to this "
         "question. First decide which single tool is needed, use ONLY that tool and "
         "ignore all the others, then answer.\n\n")
with open(clsf,"w") as c, open(segf,"w") as s, open(routef,"w") as rt:
    for l in open(src):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        (c if r.get("probe_type")=="CLS" else s).write(l+"\n")
        r2=dict(r); r2["question"]=INSTR+r2["question"]; rt.write(json.dumps(r2)+"\n")
print("split + self_route built")
PY

if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype float16 --seed "$SEED" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_dissoc_p${PORT}.log" 2>&1 &
  VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VP 2>/dev/null || { echo "vLLM DIED"; tail -20 "$DUCK/logs/vllm_dissoc_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

run() {  # run <prefix> <condition> <data> [extra...]
  local prefix="$1" cond="$2" data="$3"; shift 3
  echo "=== $prefix ($cond) ==="
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$data" --device cuda --log-prefix "$prefix" \
    --llm-parse --decoding greedy --seed "$SEED" --model-dtype fp16 \
    --path dissociation --condition "$cond" "$@"
}

run ds_no_tool     no_tool     "$PROBE"      --disable-tools
run ds_oracle_cls  oracle      "$CLS_FILE"   --tools "$CLS_TOOL"
run ds_oracle_seg  oracle      "$SEG_FILE"   --tools "$SEG_TOOL"
run ds_all_real    all_real    "$PROBE"      --tools "$CLS_TOOL,$SEG_TOOL"
run ds_self_route  self_route  "$ROUTE_FILE" --tools "$CLS_TOOL,$SEG_TOOL"
echo "=== DISSOCIATION DONE ==="
