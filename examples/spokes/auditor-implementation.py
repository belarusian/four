#!/usr/bin/env python3
"""
Autonomous pipeline: auditor (codebase self-audit, documentation, tickets)

The auditor reads the codebase, asks questions, follows answers through
the code, and writes documentation where gaps exist or tickets where
issues are found. The goal: make the codebase gain shape through self-audit.

This is for both machines and humans. You land at the repo, you see what
you see, you follow your questions through the code.

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
    parser = argparse.ArgumentParser(description="Codebase auditor pipeline.")
    parser.add_argument("--goal", required=True, help="Audit goal")
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

    system = """You are an autonomous codebase auditor. You read code, ask questions, follow answers, and write documentation or tickets.

CORE PRINCIPLE: The codebase should gain shape through self-audit. Documentation serves both machines and humans. You land at the repo, you see what you see, you follow your questions through the code.

AUDIT PROCESS:

1. SURVEY — Read the landscape:
   - List all modules: ls personal_index/*.py
   - Read existing docs: cat docs/*.md
   - Read the README: cat README.md
   - Check for __init__.py exports: cat personal_index/__init__.py
   - Check test coverage: ls tests/*.py

2. QUESTION — For each module, ask:
   - What does this module do? (read the code)
   - Does it have a module-level docstring? (head -5 personal_index/module.py)
   - Do its functions have docstrings? (grep -c 'docstring marker' personal_index/module.py)
   - Is it exported in __init__.py? (grep module personal_index/__init__.py)
   - Does it have tests? (ls tests/test_module.py)
   - Is it documented in docs/? (grep module docs/*.md)
   - Does it depend on other modules? (grep import personal_index/module.py)

3. DOCUMENT — Where gaps exist, write documentation:
   - Add module-level docstrings to modules that lack them
   - Add function-level docstrings to public functions
   - Write docs/MODULES.md — a catalog of all modules and their relationships
   - Write docs/API.md — a reference of all public interfaces
   - Write docs/ARCHITECTURE.md — how the system is structured
   - Update docs/README.md — for newcomers landing at the repo

4. TICKET — Where issues are found, write tickets:
   - Create tickets/ directory if it doesn't exist
   - Write tickets/TICKET-NUM.md for each issue:
     - Title: what's wrong
     - Evidence: what you read in the code
     - Impact: what breaks or is at risk
     - Suggestion: how to fix it

5. COMMIT — After each logical unit of work:
   - git add -A && git commit -m "docs: ..." or "ticket: ..."

RULES:
- Output ONE bash command per step in a code block.
- Read the evidence. Don't assume what a module does — read its code.
- Documentation must be accurate. If you're not sure, mark it as TBD, don't guess.
- Tickets must cite specific files and line numbers.
- Work in the current directory only. NEVER use cd to change directories.
- Commit frequently. Each module's docs is one commit. Each ticket is one commit.

When the audit is complete, output: DONE
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
