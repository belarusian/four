#!/usr/bin/env python3
"""
Autonomous pipeline: PR review and merge (reads open PRs, verifies, merges)

Reviews open PRs by reading their commits, running tests, and merging approved ones.
This is Phase 3 of the dual pipeline:
    Phase 1: Stream (commits grow out) — PROVEN
    Phase 2: PR consolidation (reads anchors, creates PRs) — PROVEN
    Phase 3: Review (reads PRs, merges to main) — TESTING NOW
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import context_aware_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory


def main():
    parser = argparse.ArgumentParser(description="PR review and merge pipeline.")
    parser.add_argument("--goal", required=True, help="Review goal (e.g., 'review all open PRs')")
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

    system = """You are an autonomous PR review and merge agent.

CORE PRINCIPLE: Read each PR, verify the code works, merge if good.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Work in the current directory only. NEVER use cd to change directories.

REVIEW APPROACH:
1. List open PRs: gh pr list --state open
2. For each PR:
   a. Read details: gh pr view <NUM> --json title,body,commits,files
   b. Checkout the PR branch: gh pr checkout <NUM>
   c. Run tests: python -m pytest tests/ -v --tb=short -q 2>&1 | tail -20
   d. If tests pass: gh pr merge <NUM> --squash --delete-branch
   e. If tests fail: gh pr comment <NUM> --body "Tests failed: ..."
   f. Return to main: git checkout main
3. Track which PRs have been processed
4. If a PR has no tests or trivial changes, approve and merge directly

EXAMPLE COMMANDS:
- gh pr list --state open: List open PRs
- gh pr view <NUM> --json title,body,commits,files: Read PR details
- gh pr checkout <NUM>: Checkout PR branch
- python -m pytest tests/ -v --tb=short -q: Run tests
- gh pr merge <NUM> --squash --delete-branch: Merge PR with squash
- gh pr comment <NUM> --body "...": Comment on PR
- git checkout main: Return to main

IMPORTANT: Be thorough but efficient. Run the full test suite for each PR.
Only merge if tests pass. Comment with specific failures if they don't.

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
