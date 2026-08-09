#!/usr/bin/env python3
"""
Autonomous pipeline: documentation (reads git history, writes docs)

Post-hoc documentation pipeline that scans the git history and generates
comprehensive documentation for the codebase.

DISTINCTION from stream-implementation:
    - Reads existing code and git history as evidence
    - Generates README, module docs, API docs, architecture docs
    - Cleans up inconsistencies, adds missing documentation
    - Additive: creates docs alongside code, never mutates existing files

This spoke implements a documentation loop using four.core.run().
The agent uses bash to read code, analyze structure, and write docs.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from four.core import run, Ok, Err
from four.chat_model import context_aware_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory


def main():
    parser = argparse.ArgumentParser(description="Documentation pipeline that reads git history and writes docs.")
    parser.add_argument("--goal", required=True, help="The documentation goal")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps")
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    LARGE_MODEL = os.getenv("FIVE_LARGE_MODEL", "qwen")
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://192.168.1.157:8080/v1")
    LARGE_URL = os.getenv("FIVE_LARGE_URL", "http://192.168.1.161:8081/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "65536"))

    step_num = [0]

    invoke = context_aware_invoke(
        fast_model=f"openai/{MODEL_ID}",
        large_model=f"openai/{LARGE_MODEL}",
        fast_base_url=BASE_URL,
        large_base_url=LARGE_URL,
        context_limit=50_000,
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        api_key="dummy",
    )

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = invoke(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    system = """You are an autonomous documentation agent.

CORE PRINCIPLE: Read the evidence, write the docs. The git history is your source of truth.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Take small incremental steps. Do NOT try to do everything in one command.
- Use git frequently: git add, git commit after each logical unit of work.
- Commit messages must describe WHAT changed and WHY.
- Continue until the documentation goal is fully achieved.
- Work in the current directory only. NEVER use cd to change directories.

DOCUMENTATION APPROACH:
1. Read the git history to understand what was built and why
2. Read the code to understand the structure and relationships
3. Write comprehensive documentation:
   - README.md: Project overview, installation, usage
   - docs/ARCHITECTURE.md: System design, module relationships
   - docs/API.md: API reference for all modules
   - docs/MODULES.md: Per-module documentation
   - docs/CHANGELOG.md: Generated from git history
4. Add inline documentation to code files where missing
5. Clean up inconsistencies, but never delete working code

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
