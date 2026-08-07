#!/usr/bin/env python3
import argparse
import os
import time

from four.core import run, Ok, Err
from four.chat_model import litellm_toolcall_invoke
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = litellm_toolcall_invoke(
            model=f"openai/{MODEL_ID}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
            tools=tools,
        )(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = str(result.value)[:120].replace("\\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    v1 = toolcall_parse()
    v2 = local_env()

    prompt = args.prompt or "List all .py files in the current directory."

    def emit(messages, outcome):
        import json
        from pathlib import Path
        path = Path("trajectory.json")
        path.write_text(json.dumps({"outcome": outcome, "messages": messages}, indent=2))
        return path

    path = run(G=debug_g, V1=v1, V2=v2, emit=emit, system="", prompt=prompt, max_steps=MAX_STEPS)
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
