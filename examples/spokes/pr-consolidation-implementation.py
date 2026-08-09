#!/usr/bin/env python3
"""
Autonomous pipeline: PR consolidation (reads git history, creates PRs)

Post-hoc PR pipeline that scans the git history and creates semantic PRs
to consolidate related commits.

DISTINCTION from stream-implementation:
    - Reads existing git history as evidence
    - Groups related commits into semantic PRs
    - Creates PRs with meaningful titles and descriptions
    - Additive: creates PRs from existing commits, never mutates history

This spoke implements a PR consolidation loop using four.core.run().
The agent uses bash to read git history, group commits, and create PRs.
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
    parser = argparse.ArgumentParser(description="PR consolidation pipeline that reads git history and creates semantic PRs.")
    parser.add_argument("--goal", required=True, help="The PR consolidation goal")
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

    system = """You are an autonomous PR consolidation agent.

CORE PRINCIPLE: Read the git history, create semantic PRs. The commits are your evidence.

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Take small incremental steps. Do NOT try to do everything in one command.
- Work in the current directory only. NEVER use cd to change directories.

PR CONSOLIDATION APPROACH:
1. Read the git history to understand what was built
2. Group related commits into semantic batches:
   - Feature commits: related functionality
   - Fix commits: bug fixes and improvements
   - Test commits: test additions and updates
   - Doc commits: documentation changes
3. For each batch, create a PR:
   - Create a branch from the batch's base commit
   - Cherry-pick the related commits
   - Push the branch
   - Create a PR with meaningful title and description
4. Track which commits have been PR'd to avoid duplicates

EXAMPLE COMMANDS:
- git log --oneline -50: Read recent commits
- git show <hash>: Read a specific commit
- git checkout -b pr/<feature> <base>: Create PR branch
- git cherry-pick <hash>: Add commit to branch
- git push origin pr/<feature>: Push branch
- gh pr create --base main --head pr/<feature> --title "..." --body "...": Create PR

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
