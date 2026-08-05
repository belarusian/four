# Live Test Results

Tested Four against multiple local models via llama.cpp (OpenAI-compatible endpoint).

## Usage

Run live tests with `examples/run_live.py`:

```bash
# Chat Completions API (regex parsing)
FIVE_BASE_URL=http://192.168.1.161:8082/v1 python examples/run_live.py --variant chat

# Tool calls API
FIVE_BASE_URL=http://192.168.1.157:8080/v1 python examples/run_live.py --variant toolcall

# Responses API (direct HTTP /v1/responses)
FIVE_BASE_URL=http://192.168.1.161:8082/v1 python examples/run_live.py --variant responses
```

Environment variables:
- `FIVE_MODEL` — Model ID (default: `fast-qwen`)
- `FIVE_BASE_URL` — llama.cpp endpoint (e.g., `http://192.168.1.157:8080/v1`)
- `FIVE_MAX_TOKENS` — Max tokens per response (default: 1024)
- `FIVE_MAX_STEPS` — Max loop steps (default: 10)

Each run saves a trajectory to `examples/log_<endpoint>_<variant>.json`.

## Results

### Qwen3-Coder-Next (80B, Q4, 17GB) — ✅ Excellent
- **Endpoint:** `http://192.168.1.161:8082/v1`
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
