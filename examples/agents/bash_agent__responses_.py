#!/usr/bin/env python3
import argparse
import os
import time

from four.core import run, Ok, Err
from four.response_model import http_response_invoke
from four.parse import toolcall_parse
from four.env import local_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    BASE_URL = args.endpoint
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))
    MAX_STEPS = int(os.getenv("FIVE_MAX_STEPS", "10"))

    step_num = [0]

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
            preview = result.value[:120].replace("\\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    v1 = toolcall_parse()
    v2 = local_env()

    system = (
        "You are a bash agent. You solve tasks by executing bash commands. "
        "Wrap each command in a ```bash ... ``` block. "
        "When the task is fully done, respond with plain text instead of a code block."
    )

    prompt = args.prompt or "List all .py files in the current directory."

    def emit(messages, outcome):
        import json
        from pathlib import Path
        path = Path("trajectory.json")
        path.write_text(json.dumps({"outcome": outcome, "messages": messages}, indent=2))
        return path

    path = run(G=debug_g, V1=v1, V2=v2, emit=emit, system=system, prompt=prompt, max_steps=MAX_STEPS)
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
