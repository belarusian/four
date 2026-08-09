#!/usr/bin/env python3
"""
Autonomous pipeline: PR consolidation (reads save anchors, creates semantic PRs)

Post-hoc PR pipeline that scans the git history for "autonomous: save progress"
anchors, groups related commits between them, and creates semantic PRs.

This is Phase 2 of the dual pipeline:
    Phase 1: Stream (commits grow out) — PROVEN
    Phase 2: PR consolidation (reads anchors, creates PRs) — TESTING NOW
    Phase 3: Review (reads PRs, merges to main) — FUTURE
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
    parser = argparse.ArgumentParser(description="PR consolidation pipeline.")
    parser.add_argument("--goal", required=True, help="Description of what to consolidate")
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

    system = """You are an autonomous PR consolidation agent.

CORE PRINCIPLE: Read the git history, find semantic groups, create PRs.
The "autonomous: save progress" commits are anchors — they mark batch boundaries.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Work in the current directory only. NEVER use cd to change directories.

CONSOLIDATION APPROACH:
1. Find all "autonomous: save progress" anchors: git log --oneline | grep "autonomous: save progress"
2. For each pair of consecutive anchors, read the commits between them
3. Group semantically related commits into batches:
   - Related feature commits go together
   - Test commits go with their feature commits
   - Fix commits go with what they fix
4. For each semantic batch, create a PR:
   a. git checkout -b pr/<short-name> <base-anchor-hash>
   b. git cherry-pick <hash1> <hash2> ... (the related commits)
   c. git push origin pr/<short-name>
   d. gh pr create --base main --head pr/<short-name> --title "<descriptive title>" --body "<what was built, why, commits>"
   e. git checkout main
5. Track created PRs to avoid duplicates
6. Skip anchors that have no meaningful commits between them

EXAMPLE COMMANDS:
- git log --oneline | grep "autonomous: save progress": Find anchors
- git log --oneline <hash1>..<hash2>: Commits between anchors
- git show <hash>: Details of a commit
- git checkout -b pr/feature <base>: Create PR branch from base
- git cherry-pick <hash>: Add commit to branch
- git push origin pr/feature: Push branch
- gh pr create --base main --head pr/feature --title "..." --body "...": Create PR
- gh pr list: Check existing PRs

IMPORTANT: Each PR should represent a coherent unit of work.
Don't create PRs for single commits or trivial changes.
Group by semantic meaning, not by anchor boundaries.

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
