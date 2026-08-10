#!/usr/bin/env python3
"""
Autonomous pipeline: PR fix (reads failing PRs, fixes issues, re-tests)

Phase 4 of the pipeline:
    Phase 1: Stream (commits grow out) — PROVEN
    Phase 2: PR consolidation (reads anchors, creates PRs) — PROVEN
    Phase 3: Review (reads PRs, merges if good, holds if broken) — PROVEN
    Phase 4: Fix (reads held PRs, fixes issues, re-tests, merges) — NEW

This spoke reads open PRs that have failing tests, checks out the branch,
fixes the issues, re-runs tests, and merges if they pass.
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
    parser = argparse.ArgumentParser(description="PR fix pipeline.")
    parser.add_argument("--goal", required=True, help="Fix goal (e.g., 'fix all failing PRs')")
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

    system = """You are an autonomous PR fix agent.

CORE PRINCIPLE: Read failing PRs, fix the issues, re-test, merge.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Work in the current directory only. NEVER use cd to change directories.

FIX APPROACH:
1. List open PRs: gh pr list --state open
2. For each PR:
   a. Checkout: gh pr checkout <NUM>
   b. Run targeted tests for the changed modules
   c. If tests fail, read the error, understand the issue, fix the code
   d. Re-run tests until they pass
   e. Commit the fix: git add -A && git commit -m "fix: <description>"
   f. Push: git push origin <branch> --force-with-lease
   g. Merge: gh pr merge <NUM> --squash --delete-branch
   h. Return to main: git checkout main
3. If a PR cannot be fixed (missing dependencies, fundamental design issue), close it with gh pr close <NUM>

EXAMPLE COMMANDS:
- gh pr list --state open: List open PRs
- gh pr checkout <NUM>: Checkout PR branch
- python -m pytest tests/test_<module>.py -v: Run targeted tests
- git add -A && git commit -m "fix: ...": Commit fix
- git push origin <branch> --force-with-lease: Push fix
- gh pr merge <NUM> --squash --delete-branch: Merge PR
- gh pr close <NUM>: Close unfixable PR
- git checkout main: Return to main

IMPORTANT: Be thorough. Fix the root cause, not just the symptom.
Run tests after every fix to verify.

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
