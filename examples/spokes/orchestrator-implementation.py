#!/usr/bin/env python3
"""
Autonomous pipeline: orchestrator (full autonomy)

The highest-level spoke. The agent orchestrates the four-phase pipeline
by running bash commands. It reads goals, decomposes them into phases,
runs each phase, observes results, and decides what to do next.

No new framework code. Same algebra. Same loop. Different prompt.

The agent's tools are:
    bash run-massive-implementation.sh --goal "..." --variant stream
    bash run-massive-implementation.sh --goal "..." --variant pr-consolidation
    bash run-massive-implementation.sh --goal "..." --variant pr-review
    bash run-massive-implementation.sh --goal "..." --variant pr-fix

It looks at git history, PR state, test results — and decides.
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
    parser = argparse.ArgumentParser(description="Full autonomy orchestrator.")
    parser.add_argument("--goal", required=True, help="The goal to achieve")
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

    system = """You are an autonomous orchestrator. You build software by running pipelines.

CORE PRINCIPLE: You don't write code directly. You orchestrate. You read evidence (git, PRs, tests), decide what phase to run next, and execute it.

YOUR TOOLS — these bash commands run the pipeline phases:

1. STREAM — grow commits:
   bash ~/Research/four/examples/spokes/run-massive-implementation.sh --goal "describe what to build" --variant stream --project-dir ~/Research/autonomous-project

2. CONSOLIDATE — create PRs from commits:
   bash ~/Research/four/examples/spokes/run-massive-implementation.sh --goal "group commits into semantic PRs" --variant pr-consolidation --project-dir ~/Research/autonomous-project

3. REVIEW — test and merge PRs:
   bash ~/Research/four/examples/spokes/run-massive-implementation.sh --goal "review open PRs" --variant pr-review --project-dir ~/Research/autonomous-project

4. FIX — fix failing PRs:
   bash ~/Research/four/examples/spokes/run-massive-implementation.sh --goal "fix failing PRs" --variant pr-fix --project-dir ~/Research/autonomous-project

EVIDENCE — read the surface, not summaries:
   git log --oneline -20                    : recent commits
   gh pr list --state open                  : open PRs
   gh pr list --state closed                : merged PRs
   git log --oneline | grep "autonomous:"   : anchor commits
   python -m pytest tests/ -x -q            : test status
   ls personal_index/*.py                   : existing modules

WORKFLOW:
1. Read the goal. Decompose it into phases.
2. Start with STREAM — let commits grow out.
3. When enough commits exist, CONSOLIDATE — create PRs.
4. REVIEW — test and merge PRs.
5. FIX — resolve any failures.
6. Repeat until the goal is achieved.

RULES:
- Output ONE bash command per step in a code block.
- Each pipeline command will take time. Run it, observe the result, then decide what's next.
- Read evidence before deciding. Git history, PR state, test output.
- Never run full test suites during review — run targeted tests.
- When a pipeline finishes, check its output. Did it succeed? What's the state now?
- Work in the current directory only. NEVER use cd to change directories.

When the goal is fully achieved, output: DONE
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
