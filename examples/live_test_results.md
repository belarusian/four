# Live Test Results

Tested Four against multiple local models via llama.cpp (OpenAI-compatible endpoint).

## Test Script

```python
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from four.core import run, Ok, Err
from four.model import retry_invoke, litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory

MODEL_ID = '/path/to/model.gguf'
BASE_URL = "http://192.168.1.XXX:8080/v1"
LITELLM_MODEL = f"openai/{MODEL_ID}"

step_num = [0]

def debug_g(messages):
    step_num[0] += 1
    t0 = time.time()
    result = litellm_invoke(
        model=LITELLM_MODEL,
        base_url=BASE_URL,
        temperature=0.3,
        max_tokens=1024,
        api_key="dummy",
    )(messages)
    elapsed = time.time() - t0
    if isinstance(result, Ok):
        preview = result.value[:120].replace("\n", " ")
        print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
    else:
        print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
    return result

g = retry_invoke(debug_g, max_attempts=3)
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

path = run(G=g, V1=v1, V2=v2, emit=emit, system=system, prompt=prompt, max_steps=10)
print(f"\nTrajectory saved to: {path}")
print(f"Total G calls: {step_num[0]}")
```

## Results

### Qwen3-Coder-Next (80B, Q4, 17GB) — ✅ Excellent
- **Endpoint:** `http://192.168.1.161:8080/v1`
- **4 steps** to solve "list .py files, count lines in largest"
- Correct answer: 655 lines in `generate_projections.py`
- Clean termination on `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- Avg latency: 1.1s, no reasoning overhead
- **Verdict: Production-ready**

### Qwen3-Coder-Next (80B, Q8, 48GB) — ✅ Excellent
- **Endpoint:** `http://192.168.1.161:8082/v1`
- **4 steps**, correct answer: 653 lines
- Clean termination
- Avg latency: 1.2s (same as Q4)
- **Verdict: Q8 offers no measurable benefit over Q4**

### Qwen3.6-35B-A3B MoE (Q8, 39GB) — ⚠️ Decent
- **Endpoint:** `http://192.168.1.161:8081/v1`
- 7 steps to solve the same task
- Wrong answer: 653 (total lines, not largest file)
- Clean termination
- Avg latency: 2.0s
- **Verdict: Terminates correctly but weaker at planning**

### Qwen3.6-27B (Q4, 17GB) — ❌ Stuck
- **Endpoint:** `http://192.168.1.157:8080/v1`
- Hit max_steps (10) — re-ran same command endlessly
- Found correct answer (394) but couldn't follow termination instruction
- Long reasoning detours (13.7s per step)
- **Verdict: Too small for self-termination**

### Responses API variants vs Chat Completions

Tested both paths against Qwen3.6-27B (the "quick-qwen" model on `:8080`):

| Variant | Result |
|---------|--------|
| Chat Completions (`litellm.completion`) | Stuck at 10 steps |
| Responses API (`litellm.responses`) | Auth error (llama.cpp doesn't support this endpoint) |
| Direct HTTP `/v1/responses` | Stuck at 10 steps |

**Conclusion:** The API variant doesn't change model behavior. The 27B model gets stuck regardless of which G implementation is used. The issue is model capability, not the client.

## Fixes Applied During Testing

1. **`model.py`:** Added `reasoning_content` fallback when `content` is empty (Qwen3 outputs into `reasoning_content`, not `content`)
2. **`parse.py`:** Added single-line code block matching (`bash cmd` not just `bash\n...\n`) — Coder-Next puts commands on same line as backtick marker
3. **`litellm_invoke`:** `openai/` prefix required for llama.cpp endpoints

## Notes

- llama.cpp needs `openai/` prefix in model name for litellm to route correctly
- `extra_body={"reasoning": {"enabled": false}}` helps reduce reasoning overhead but Coder-Next doesn't need it
- The exit signal detection in `env.py` works correctly — the 27B model just never reached it
- API key can be set to anything for llama.cpp (it does not require auth)
- Q4 vs Q8 quantization makes no measurable difference in performance
