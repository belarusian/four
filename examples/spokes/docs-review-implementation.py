#!/usr/bin/env python3
"""
Autonomous pipeline: docs review (checks doc completeness, adds missing docs)

Reviews the existing documentation and identifies gaps. Generates minimal
missing docs to achieve 100% coverage.

DISTINCTION from documentation-implementation:
    - Not creating docs from scratch
    - Auditing existing docs and filling gaps
    - Minimal, focused additions

This spoke implements a docs audit loop using four.core.run().
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
    parser = argparse.ArgumentParser(description="Docs review pipeline.")
    parser.add_argument("--goal", required=True, help="Documentation review goal")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps")
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

    system = """You are an autonomous documentation reviewer.

CORE PRINCIPLE: Audit existing docs, identify gaps, add minimal coverage.

DOCS TO AUDIT:
- README.md (must exist with: overview, install, usage)
- docs/ARCHITECTURE.md (system design, module map)
- docs/API.md (all public functions/classes documented)
- docs/MODULES.md (per-module purpose and key exports)
- docs/CHANGELOG.md (generated from git history)

AUDIT APPROACH:
1. List all Python modules in personal_index/
2. For each module, check if it has docstring or docs entry
3. If missing: write minimal doc (module purpose, key exports, example)
4. Cross-reference modules to ensure relationships are documented
5. Ensure README links to all relevant docs

RULES:
- Output ONE bash command per step in a code block.
- Create docs ONLY if they don't exist or are incomplete.
- DO NOT rewrite existing working docs.
- Use git add/commit after each doc addition.
- Commit message: "docs: add missing docs for X"

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
