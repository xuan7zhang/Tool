#!/bin/bash
# Exp D: does distractor pollution degrade the confidence->correctness signal?
# CLS probe + classifier + {0,5,20} distractors, WITH per-token logprobs, so we can
# measure AUC(msg_last -> correct) per pollution level. Serves one vLLM, 3 conditions.
set -o pipefail
export DS=/datasets/omni_pretraining
export HF_HOME=$DS/.hf_home HF_HUB_CACHE=$DS/.hf_home/hub TRANSFORMERS_CACHE=$DS/.hf_home/hub
export HF_ASSETS_CACHE=$DS/.hf_home/assets HF_DATASETS_CACHE=$DS/.hf_datasets   # scratch over quota
export DUCK=${DUCK:-/project/aip-xli135/xzhan576/DUCK}
export MEDRAX_WEIGHTS=$DS/.medrax_weights MEDRAX_TEMP=$DS/.medrax_temp
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
VLLM_PY=$DS/.venvs/vllm/bin/python; DUCK_PY=$DS/.venvs/DUCK/bin/python
SERVED_NAME=qwen3-vl-8b; MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
PORT=${PORT:-8000}; VLLM_GPU=${VLLM_GPU:-0}; TOOL_GPU=${TOOL_GPU:-1}
PROBE=${PROBE:-$DUCK/logs/toolbias/dissociation/probe_cls.jsonl}
RESULTS=$DUCK/logs/toolbias/confidence; mkdir -p "$RESULTS"
cd "$DUCK"

export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --served-model-name "$SERVED_NAME" --port $PORT \
  --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype float16 --seed 0 \
  --max-num-seqs 64 --enable-auto-tool-choice --tool-call-parser hermes \
  > "$DUCK/logs/vllm_conf_p${PORT}.log" 2>&1 &
VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
  kill -0 $VP 2>/dev/null || { echo "vLLM DIED"; tail -20 "$DUCK/logs/vllm_conf_p${PORT}.log"; exit 1; }
  sleep 5
done
export OPENAI_BASE_URL="http://localhost:$PORT/v1"; export OPENAI_API_KEY=EMPTY

run() {  # run <prefix> <cond> [extra...]
  local prefix="$1" cond="$2"; shift 2
  echo "=== $prefix ($cond) ==="
  CUDA_VISIBLE_DEVICES=$TOOL_GPU HF_HOME=$MEDRAX_WEIGHTS HF_DATASETS_CACHE=$DS/.hf_datasets $DUCK_PY launch_over_chexbench.py \
    --model "$SERVED_NAME" --model-dir "$MEDRAX_WEIGHTS" --temp-dir "$MEDRAX_TEMP" \
    --data-file "$PROBE" --device cuda --log-prefix "$prefix" \
    --llm-parse --decoding greedy --seed 0 --model-dtype fp16 --capture-logprobs 1 \
    --tools ChestXRayClassifierTool \
    --path confidence --condition "$cond" "$@" || echo "$prefix FAILED"
}

run conf_clean clean
for K in 5 20; do
  cfg="$RESULTS/distract${K}.yaml"
  $DUCK_PY - "$cfg" "$K" <<'PY'
import sys
from ducx_noise.config import NoiseConfig, DistractorConfig
NoiseConfig(seed=0, tool_order="shuffle", label=f"conf{sys.argv[2]}",
  distractor=DistractorConfig(enabled=True, count=int(sys.argv[2]), position="random", similarity="obvious")).save(sys.argv[1])
PY
  run "conf_pol${K}" "pol${K}" --noise-config "$cfg"
done
echo "=== CONFIDENCE POLLUTION DONE ==="
