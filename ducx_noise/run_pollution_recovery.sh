#!/bin/bash
# Many-distractor regime: does a polluted toolbox (2 real + K distractors) hurt vs
# the pruned oracle (classifier only), on the SAME 225 CLS questions? Retrieval is
# perfect (top1=1.0 with 10 distractors) so pruned == oracle; the recovery test is
# oracle vs polluted, paired McNemar. Reuses ds_oracle_cls from run_dissociation.
set -eo pipefail
export DS=/datasets/omni_pretraining
export HF_HOME=${HF_HOME:-$DS/.hf_home}
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=${MEDRAX_WEIGHTS:-$DS/.medrax_weights}
export MEDRAX_TEMP=${MEDRAX_TEMP:-$DS/.medrax_temp}
VLLM_PY=$DS/.venvs/vllm/bin/python; DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b; MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}; VLLM_GPU=${VLLM_GPU:-0}; TOOL_GPU=${TOOL_GPU:-1}; SEED=${SEED:-0}
CLS_FILE=${CLS_FILE:-$DUCK/logs/toolbias/dissociation/probe_cls.jsonl}
RESULTS=${RESULTS:-$DUCK/logs/toolbias/pollution}; mkdir -p "$RESULTS"
COUNTS=${COUNTS:-"5 10 20"}
cd "$DUCK"

if [ -z "$OPENAI_BASE_URL" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype float16 --seed "$SEED" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$DUCK/logs/vllm_pollution_p${PORT}.log" 2>&1 &
  VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
    kill -0 $VP 2>/dev/null || { echo "vLLM DIED"; tail -20 "$DUCK/logs/vllm_pollution_p${PORT}.log"; exit 1; }
    sleep 5
  done
  export OPENAI_BASE_URL="http://localhost:$PORT/v1"
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

for K in $COUNTS; do
  cfg="$RESULTS/distract${K}.yaml"
  $DUCK_PY - "$cfg" "$K" <<'PY'
import sys
from ducx_noise.config import NoiseConfig, DistractorConfig
NoiseConfig(seed=0, tool_order="shuffle", label=f"pollute{sys.argv[2]}",
    distractor=DistractorConfig(enabled=True, count=int(sys.argv[2]),
        position="random", similarity="obvious")).save(sys.argv[1])
PY
  echo "=== polluted_${K} (2 real + ${K} distractors) ==="
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$CLS_FILE" --device cuda --log-prefix "pol_${K}" \
    --llm-parse --decoding greedy --seed "$SEED" --model-dtype fp16 \
    --path pollution --condition "polluted_${K}" \
    --tools ChestXRayClassifierTool,ChestXRaySegmentationTool --noise-config "$cfg"
done
echo "=== POLLUTION RECOVERY DONE ==="
