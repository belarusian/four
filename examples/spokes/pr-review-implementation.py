#!/usr/bin/env python3
"""
Autonomous pipeline: PR review (reads PR, verifies code, approves/rejects)

Reviews a PR by reading its commits, running tests, and verifying code quality.
This is the second half of the dual pipeline: stream → PR → review → merge.
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
    parser = argparse.ArgumentParser(description="PR review pipeline that verifies code and approves/rejects.")
    parser.add_argument("--pr", required=True, help="PR number to review")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
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

    system = """You are an autonomous PR review agent.

CORE PRINCIPLE: Read the evidence, verify the code. The PR is your source of truth.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Work in the current directory only. NEVER use cd to change directories.

REVIEW APPROACH:
1. Fetch the PR: gh pr view <PR_NUM> --json title,body,commits,files
2. Read the commits and changed files
3. Run tests: python -m pytest tests/ -v
4. Run linters: flake8 personal_index/ (if available)
5. Verify code quality:
   - Tests pass
   - No regressions
   - Code follows project conventions
   - Documentation is adequate
6. If everything passes: gh pr review <PR_NUM> --approve --body "Reviewed and approved"
7. If there are issues: gh pr review <PR_NUM> --comment --body "Issues found: ..."

When done, output: DONE
"""

    path = run(
        G=debug_g,
        V1=regex_parse(),
        V2=local_env(),
        emit=save_trajectory(),
        system=system,
        prompt=f"Review PR #{args.pr}. Read the commits, run tests, verify code quality. Approve if good, comment with issues if not.",
        max_steps=args.max_steps,
    )
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
