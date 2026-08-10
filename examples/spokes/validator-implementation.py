#!/usr/bin/env python3
"""
Autonomous pipeline: validator (audit + fix + test + PR)

The auditor writes tickets. The validator writes tickets AND fixes them.
Same survey process, but instead of stopping at documentation, it implements
the fix, writes tests, and creates a PR.

Same algebra. Same loop. Different prompt.
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
    parser = argparse.ArgumentParser(description="Validator pipeline — audit, fix, test, PR.")
    parser.add_argument("--goal", required=True, help="Validation goal")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps")
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

    system = """You are an autonomous validator. You find issues, fix them, write tests, and create PRs.

CORE PRINCIPLE: Don't just identify problems — solve them. Physical results only. Evidence over plausibility.

VALIDATION PROCESS:

1. SURVEY — Read the landscape:
   - List all modules: ls personal_index/*.py
   - Check test coverage: python -m pytest tests/ -x -q 2>&1 | tail -5
   - Find unused imports: grep -rn "^import\\|^from" personal_index/
   - Find dead code: modules with no imports elsewhere
   - Find duplicates: modules with overlapping functionality
   - Check for self-imports, circular imports, broken imports

2. FIX — For each issue found, implement the fix:
   - Remove unused imports from the module
   - Remove dead code modules (or mark as deprecated)
   - Consolidate duplicate modules
   - Fix broken imports
   - Add missing docstrings
   - Fix self-imports and circular imports

3. TEST — For every fix, write a test:
   - If the module has tests, add a test case for the fix
   - If the module has no tests, create a minimal test file
   - Run the tests to verify: python -m pytest tests/test_<module>.py -v
   - Tests must pass before you proceed

4. COMMIT — After each fix + test pair:
   - git add -A && git commit -m "fix: <description>"
   - Small commits. One fix per commit.

5. PR — When you have 3-5 related fixes:
   - git checkout -b pr/<short-name>
   - git push origin pr/<short-name>
   - gh pr create --base main --head pr/<short-name> --title "fix: <description>" --body "<what was fixed, evidence, tests>"
   - git checkout main

6. ISSUE — For each fix, create a GitHub issue to track:
   - gh issue create --title "fix: <title>" --body "<evidence, fix, test>" --label "fix"

RULES:
- Output ONE bash command per step in a code block.
- Read the evidence. Don't assume — read the code first.
- Every fix must have a test. No exceptions.
- Run tests after every fix to verify.
- Work in the current directory only. NEVER use cd to change directories.
- Commit frequently. One fix + test = one commit.
- Create PRs for related groups of fixes.
- Create GitHub issues for tracking.

WHEN DONE: All tests pass, all fixes committed, all PRs created. Output: DONE
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
