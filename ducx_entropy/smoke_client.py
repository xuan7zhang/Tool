"""Smoke probe for the reasoning-entropy task (Instruct + path A).

Reasoning segment = the model's final-answer prose (the terminal assistant
message). We re-run rollouts with logprobs enabled and compute per-token
likelihood/entropy over that text. This probe verifies the two things the
capture hook depends on:

  (1) RAW vLLM (OpenAI client): per-token logprobs come back together with a
      final text answer AND with tool_calls (hermes), for Qwen3-VL-8B-Instruct.
  (2) LANGCHAIN ChatOpenAI(logprobs=True, top_logprobs=K).bind_tools(): the
      logprobs survive into response.response_metadata -- that is exactly what
      launch_over_chexbench.py's serialize step will persist.

Run AFTER the vLLM server is up (see smoke_test.sh). Dumps the raw response to
/tmp/ducx_smoke_response.json and prints a structural summary so we can write
ducx_entropy against the real token layout.
"""
import base64
import json
import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("SMOKE_MODEL", "qwen3-vl-8b")
IMG = os.environ.get("SMOKE_IMAGE", "figures/10009/figure_1.jpg")
TOPK = int(os.environ.get("SMOKE_TOPK", "20"))
OUT = os.environ.get("SMOKE_OUT", "/tmp/ducx_smoke_response.json")

# ---------------------------------------------------------------------------
# Probe 1: raw OpenAI client -- does vLLM return logprobs + tool_calls together?
# ---------------------------------------------------------------------------
client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

with open(IMG, "rb") as fh:
    b64 = base64.b64encode(fh.read()).decode("utf-8")

tools = [
    {
        "type": "function",
        "function": {
            "name": "chest_xray_classifier",
            "description": "Classify common pathologies (e.g. consolidation, "
            "effusion, cardiomegaly) from a chest X-ray image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the X-ray."}
                },
                "required": ["image_path"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You are a chest X-ray expert. Use the provided "
     "tool when image analysis would help, then give a final answer."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Image path: {IMG}\nIs there consolidation in this "
             "chest X-ray? Use the classifier tool, then explain your answer."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    },
]

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    logprobs=True,
    top_logprobs=TOPK,
)

raw = resp.model_dump()
with open(OUT, "w") as fh:
    json.dump(raw, fh, indent=2, default=str)

choice = resp.choices[0]
msg = choice.message
content = msg.content
tool_calls = msg.tool_calls
lp = choice.logprobs
toks = lp.content if (lp and lp.content) else []

print("=" * 70)
print("PROBE 1 (raw vLLM)")
print(f"  finish_reason   : {choice.finish_reason}")
print(f"  content         : {'PRESENT' if content else 'EMPTY'}"
      f" (len={len(content) if content else 0} chars)")
print(f"  tool_calls      : {len(tool_calls) if tool_calls else 0}"
      + (f"  -> {[tc.function.name for tc in tool_calls]}" if tool_calls else ""))
print(f"  logprobs.content: {'PRESENT' if toks else 'MISSING'} ({len(toks)} tokens)")
if toks:
    print(f"  top_logprobs/tok: {len(toks[0].top_logprobs)} (requested {TOPK})")
    print("  first 8 tokens  :", [t.token for t in toks[:8]])
    print("  last 8 tokens   :", [t.token for t in toks[-8:]])
    print(f"  sample entry    : token={toks[0].token!r} logprob={toks[0].logprob:.4f}")

# ---------------------------------------------------------------------------
# Probe 2: langchain ChatOpenAI -- do logprobs survive into response_metadata?
# (This is the agent's real path; the capture hook persists this.)
# Text-only here: logprobs surfacing is modality-independent, keep it robust.
# ---------------------------------------------------------------------------
lc_ok = False
lc_logprobs_len = 0
try:
    from langchain_openai import ChatOpenAI

    lc = ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        logprobs=True,
        top_logprobs=TOPK,
    ).bind_tools(tools)
    out = lc.invoke("Name one common chest X-ray finding and briefly say why it matters.")
    rmeta = getattr(out, "response_metadata", {}) or {}
    lp2 = (rmeta.get("logprobs") or {}).get("content") or []
    lc_logprobs_len = len(lp2)
    lc_ok = lc_logprobs_len > 0
    print("=" * 70)
    print("PROBE 2 (langchain ChatOpenAI)")
    print(f"  response_metadata keys : {sorted(rmeta.keys())}")
    print(f"  logprobs.content       : {'PRESENT' if lc_ok else 'MISSING'} ({lc_logprobs_len} tokens)")
    print(f"  additional_kwargs keys : {sorted((out.additional_kwargs or {}).keys())}")
except Exception as exc:  # pragma: no cover - integration probe
    print("=" * 70)
    print(f"PROBE 2 (langchain) FAILED: {exc!r}")

print("=" * 70)
raw_ok = bool(toks)
print(f"raw response dumped -> {OUT}")
print(f"\nSMOKE RESULT: raw_logprobs={raw_ok}  tool_call_seen={bool(tool_calls)}  "
      f"langchain_surfaces_logprobs={lc_ok}")
sys.exit(0 if (raw_ok and lc_ok) else 2)
