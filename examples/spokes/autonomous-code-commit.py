#!/usr/bin/env python3
"""
Autonomous pipeline: autonomous-code-commit

Autonomous pipeline for writing code, testing, and committing changes

This spoke implements an autonomous development loop using four.core.run().
The agent uses bash to write code, run git, execute tests, and iterate.

Stages:
    - analyze: Analyze requirements, explore codebase, and draft implementation plan (max 10 turns, validate: test -f output/plan.md)
    - implement: Write code, run tests, fix issues, and iterate until all tests pass (max 10 turns, validate: make test || pytest -q || echo )
    - commit: Stage changes, write descriptive commit message, and push to remote (max 10 turns, validate: git status --porcelain | grep -q )
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory


def main():
    parser = argparse.ArgumentParser(description="Autonomous pipeline for writing code, testing, and committing changes")
    parser.add_argument("--goal", required=True, help="The goal to achieve")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps")
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "granite4.1:8b")
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "65536"))

    step_num = [0]

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = litellm_invoke(
            model=f"openai/{MODEL_ID}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    system = """You are an autonomous development agent.

Goal: Implement the user's request by writing code, committing, and pushing.

Stages to follow:
    - analyze: Analyze requirements, explore codebase, and draft implementation plan (max 10 turns, validate: test -f output/plan.md)
    - implement: Write code, run tests, fix issues, and iterate until all tests pass (max 10 turns, validate: make test || pytest -q || echo )
    - commit: Stage changes, write descriptive commit message, and push to remote (max 10 turns, validate: git status --porcelain | grep -q )

Use bash commands to:
- Write/modify files with cat, echo, or python -c
- Run git add, git commit, git push
- Execute tests, linters, builds
- Call other LLMs via curl if needed

Each step, output a bash command in a code block. The system will execute it and show you the result. Continue until the goal is achieved.

When done, output: DONE
"""

    path = run(
        G=debug_g,
        V1=regex_parse(),
        V2=local_env(),
        emit=save_trajectory(),
        system=system,
        prompt=args.goal,
        max_steps=args.max_steps,
    )
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
