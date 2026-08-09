#!/usr/bin/env python3
"""
Autonomous pipeline: context-aware (two-model fallback)

Same shape as massive-feature-implementation, but with automatic model switching.

DISTINCTION from massive-feature-implementation:
    - Uses context_aware_invoke to switch between fast and large-context models
    - Fast model (FIVE_MODEL/FIVE_BASE_URL) for normal throughput
    - Large model (FIVE_LARGE_MODEL/FIVE_LARGE_URL) when context exceeds 50k tokens
    - Automatic history compression when context exceeds 200k tokens
    - Use this variant when building projects with 100+ commits where context grows

This spoke implements an autonomous development loop using four.core.run().
The agent uses bash to write code, run git, execute tests, and iterate.

Stages:
    - setup: Initialize repository, configure CI/CD pipeline, create progress tracking spreadsheet template (max 10 turns, validate: test -f .github/workflows/ci.yml && test -f progress_tracking.csv && git rev-parse --git-dir > /dev/null 2>&1)
    - ticket1_analysis: Analyze first feature ticket requirements, decompose into 100 PRs with 5 commits each, document in plan (max 10 turns, validate: test -f output/ticket1_plan.md && grep -c )
    - ticket1_implementation: Stage: ticket1_implementation (max 10 turns, validate: true)
    - ticket2_analysis: Analyze second feature ticket requirements, decompose into 100 PRs with 5 commits each, document in plan (max 10 turns, validate: test -f output/ticket2_plan.md && grep -c )
    - ticket2_implementation: Stage: ticket2_implementation (max 10 turns, validate: true)
    - finalize: Final validation of all 1000 commits, verify 200 merged PRs, update spreadsheet with final status, generate summary report (max 10 turns, validate: test -f output/final_report.md && tail -n +2 progress_tracking.csv | wc -l | grep -q )
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
    parser = argparse.ArgumentParser(description="Context-aware autonomous pipeline with two-model fallback.")
    parser.add_argument("--goal", required=True, help="The goal to achieve")
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

    system = """You are an autonomous development agent.

Goal: Implement the user's request by writing code, committing, and pushing.

Stages to follow:
    - setup: Initialize repository, configure CI/CD pipeline, create progress tracking spreadsheet template (max 10 turns, validate: test -f .github/workflows/ci.yml && test -f progress_tracking.csv && git rev-parse --git-dir > /dev/null 2>&1)
    - ticket1_analysis: Analyze first feature ticket requirements, decompose into 100 PRs with 5 commits each, document in plan (max 10 turns, validate: test -f output/ticket1_plan.md && grep -c )
    - ticket1_implementation: Stage: ticket1_implementation (max 10 turns, validate: true)
    - ticket2_analysis: Analyze second feature ticket requirements, decompose into 100 PRs with 5 commits each, document in plan (max 10 turns, validate: test -f output/ticket2_plan.md && grep -c )
    - ticket2_implementation: Stage: ticket2_implementation (max 10 turns, validate: true)
    - finalize: Final validation of all 1000 commits, verify 200 merged PRs, update spreadsheet with final status, generate summary report (max 10 turns, validate: test -f output/final_report.md && tail -n +2 progress_tracking.csv | wc -l | grep -q )

Use bash commands to:
- Write/modify files with cat, echo, or python -c
- Run git add, git commit, git push
- Execute tests, linters, builds
- Call other LLMs via curl if needed

IMPORTANT: Work in the current directory only. NEVER use cd to change directories. NEVER create temporary directories. All files should be created in the current working directory.

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
