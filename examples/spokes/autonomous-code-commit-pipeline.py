#!/usr/bin/env python3
"""
Autonomous pipeline: autonomous-code-commit-pipeline

Autonomous pipeline that analyzes a task, plans implementation, writes code with incremental git commits, and validates via tests and linters

This spoke implements an autonomous development loop using four.core.run().
The agent uses bash to write code, run git, execute tests, and iterate.

Stages:
    - analyze: Read the task description, explore the codebase structure, identify relevant files and dependencies, and write analysis to output/analysis.md
    - plan: Create a detailed implementation plan with ordered steps, file-level changes, and commit strategy. Write to output/plan.md
    - implement: Execute the plan by writing/modifying code files, running git add and git commit after each logical change, and verifying tests pass. Use incremental commits with descriptive messages.
    - validate: Run linters, formatters, and full test suite. Fix any issues found. Ensure git status is clean and all commits are well-formed.
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


def resilient_env():
    """V2 wrapper: command failures become observations, not loop-terminating errors."""
    base = local_env()

    def _validate(command: str):
        result = base(command)
        if isinstance(result, Err):
            return Ok({"role": "tool", "content": f"<error>{result.error}</error>"})
        return result

    return _validate


def main():
    parser = argparse.ArgumentParser(description="Autonomous pipeline that analyzes a task, plans implementation, writes code with incremental git commits, and validates via tests and linters")
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
            preview = result.value[:120].replace("\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    system = """You are an autonomous development agent.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Take small incremental steps. Do NOT try to do everything in one command.
- Use git frequently: git add, git commit after each meaningful change.
- Continue until the goal is fully achieved.

Stages to follow:
    - analyze: Read the task description, explore the codebase structure, identify relevant files and dependencies, and write analysis to output/analysis.md
    - plan: Create a detailed implementation plan with ordered steps, file-level changes, and commit strategy. Write to output/plan.md
    - implement: Execute the plan by writing/modifying code files, running git add and git commit after each logical change, and verifying tests pass. Use incremental commits with descriptive messages.
    - validate: Run linters, formatters, and full test suite. Fix any issues found. Ensure git status is clean and all commits are well-formed.
"""

    path = run(
        G=debug_g,
        V1=regex_parse(),
        V2=resilient_env(),
        emit=save_trajectory(),
        system=system,
        prompt=args.goal,
        max_steps=args.max_steps,
    )
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
