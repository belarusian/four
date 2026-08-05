"""Live integration test — runs Four against a local model via llama.cpp.

Usage:
    python examples/live_test.py "your prompt here"

Environment variables:
    FIVE_MODEL       Model ID (full path to .gguf)
    FIVE_BASE_URL    llama.cpp endpoint (default: http://192.168.1.161:8080/v1)
    FIVE_MAX_TOKENS  Max tokens per response (default: 1024)
    FIVE_MAX_STEPS   Max loop steps (default: 10)
"""

import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory

MODEL_ID = os.getenv("FIVE_MODEL", "/Users/kodep/models/bartowski/qwen3-coder-next-q4/Qwen_Qwen3-Coder-Next-Q4_K_M/Qwen_Qwen3-Coder-Next-Q4_K_M.gguf")
BASE_URL = os.getenv("FIVE_BASE_URL", "http://192.168.1.161:8080/v1")
MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))
MAX_STEPS = int(os.getenv("FIVE_MAX_STEPS", "10"))
LITELLM_MODEL = f"openai/{MODEL_ID}"

step_num = [0]

def debug_g(messages):
    step_num[0] += 1
    t0 = time.time()
    result = litellm_invoke(
        model=LITELLM_MODEL,
        base_url=BASE_URL,
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        api_key="dummy",
    )(messages)
    elapsed = time.time() - t0
    if isinstance(result, Ok):
        preview = result.value[:120].replace("\n", " ")
        print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
    else:
        print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
    return result

g = debug_g
v1 = regex_parse()
v2 = local_env()
emit = save_trajectory(".")

system = (
    "You are a bash agent. You solve tasks by executing bash commands. "
    "Wrap each command in a ```bash ... ``` block. "
    "When the task is fully done, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
)

prompt = sys.argv[1] if len(sys.argv) > 1 else (
    "List all .py files in the current directory, then count how many lines are in the largest one. Show the final count."
)

print(f"Model: {MODEL_ID.split('/')[-1]}")
print(f"Endpoint: {BASE_URL}")
print(f"Prompt: {prompt[:120]}")
print("-" * 60)

path = run(G=g, V1=v1, V2=v2, emit=emit, system=system, prompt=prompt, max_steps=MAX_STEPS)
print(f"\nTrajectory saved to: {path}")
print(f"Total G calls: {step_num[0]}")
