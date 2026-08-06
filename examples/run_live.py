"""Live integration test runner against local llama.cpp models.

Usage:
    python examples/run_live.py --variant chat --endpoint http://192.168.1.161:8082/v1
    python examples/run_live.py --variant toolcall --endpoint http://192.168.1.157:8080/v1
    python examples/run_live.py --variant responses --endpoint http://192.168.1.161:8082/v1

Variants:
    chat        Chat Completions API + regex parsing (default)
    toolcall    Chat Completions API + tool calls
    responses   Responses API + tool calls

Trajectory is saved to examples/log_<endpoint>_<variant>.json

Environment variables:
    FIVE_MODEL       Model ID (default: from .env or fast-qwen)
    FIVE_MAX_TOKENS  Max tokens per response (default: 1024)
    FIVE_MAX_STEPS   Max loop steps (default: 10)
"""

import argparse
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from four.core import run, Ok, Err
from four.env import local_env
from four.core import save_trajectory


# ── Shared utilities ────────────────────────────────────────────────────────


def make_parser():
    p = argparse.ArgumentParser(description="Run Four live test")
    p.add_argument("--variant", choices=["chat", "toolcall", "responses"], default="chat")
    p.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://192.168.1.161:8082/v1"))
    p.add_argument("--prompt", default=None)
    return p


def get_system_prompt(variant):
    """Return system prompt for variant."""
    if variant in ("chat", "responses"):
        return (
            "You are a bash agent. You solve tasks by executing bash commands. "
            "Wrap each command in a ```bash ... ``` block. "
            "When the task is fully done, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        )
    return "You are a helpful assistant that can interact with a computer."


def get_litellm_model():
    """Get LITELLM_MODEL string."""
    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    return f"openai/{MODEL_ID}"


def make_debug_g_invoke(step_num, invoke_fn, MAX_TOKENS):
    """Create a debug wrapper for litellm Invoke functions."""
    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = invoke_fn(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result
    return debug_g


def make_debug_g_responses(step_num, BASE_URL, MODEL_ID, MAX_TOKENS):
    """Create a debug wrapper for http_response_invoke."""
    from four.response_model import http_response_invoke

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = http_response_invoke(
            base_url=BASE_URL,
            model=MODEL_ID,
            api_key="dummy",
            max_output_tokens=MAX_TOKENS,
        )(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result
    return debug_g


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    args = make_parser().parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    BASE_URL = args.endpoint
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))
    MAX_STEPS = int(os.getenv("FIVE_MAX_STEPS", "10"))

    step_num = [0]
    debug_g = None
    v1 = None
    system = get_system_prompt(args.variant)

    if args.variant == "chat":
        from four.chat_model import litellm_invoke

        LITELLM_MODEL = get_litellm_model()
        invoke_fn = litellm_invoke(
            model=LITELLM_MODEL,
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )
        debug_g = make_debug_g_invoke(step_num, invoke_fn, MAX_TOKENS)

        from four.parse import regex_parse
        v1 = regex_parse()

    elif args.variant == "toolcall":
        from four.chat_model import litellm_toolcall_invoke

        LITELLM_MODEL = get_litellm_model()
        invoke_fn = litellm_toolcall_invoke(
            model=LITELLM_MODEL,
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )
        debug_g = make_debug_g_invoke(step_num, invoke_fn, MAX_TOKENS)

        from four.parse import toolcall_parse
        v1 = toolcall_parse()

    elif args.variant == "responses":
        debug_g = make_debug_g_responses(step_num, BASE_URL, MODEL_ID, MAX_TOKENS)

        from four.parse import toolcall_parse
        v1 = toolcall_parse()

    # All branches above define debug_g, v1, system
    assert debug_g is not None and v1 is not None and system is not None

    v2 = local_env()

    prompt = args.prompt or (
        "List all .py files in the current directory, then count how many lines are in the largest one. Show the final count."
    )

    # Derive a clean log filename from the endpoint
    clean = BASE_URL.replace("http://", "").replace("https://", "").rstrip("/")
    if clean.endswith("/v1"):
        clean = clean[:-3]
    clean = clean.replace(":", "_")
    log_file = os.path.join(os.path.dirname(__file__), f"log_{clean}_{args.variant}.json")

    def emit(messages, outcome):
        import json
        from pathlib import Path
        p = Path(log_file)
        with open(p, "w") as f:
            json.dump({"outcome": outcome, "messages": messages}, f, indent=2)
        return p

    print(f"Variant: {args.variant}")
    print(f"Endpoint: {BASE_URL}")
    print(f"Prompt: {prompt[:120]}")
    print("-" * 60)

    path = run(G=debug_g, V1=v1, V2=v2, emit=emit, system=system, prompt=prompt, max_steps=MAX_STEPS)

    print(f"\nTrajectory saved to: {log_file}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
