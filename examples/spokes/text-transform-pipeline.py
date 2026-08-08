#!/usr/bin/env python3
"""
Autonomous pipeline: text-transform-pipeline

Pipeline that translates text to uppercase, reverses it, and counts words

This spoke implements an autonomous development loop using four.core.run().
The agent uses bash to write code, run git, execute tests, and iterate.

Stages:
    - setup: Create input text file and processing script (max 10 turns, validate: test -f input.txt && test -f transform.sh)
    - transform: Apply uppercase transformation, reverse the text, and count words (max 10 turns, validate: test -f output.txt && test -f word_count.txt)
    - verify: Verify transformations are correct: check uppercase, reversal, and word count (max 10 turns, validate: bash -c )
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory


def main():
    parser = argparse.ArgumentParser(description="Pipeline that translates text to uppercase, reverses it, and counts words")
    parser.add_argument("--goal", required=True, help="The goal to achieve")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps")
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "granite4.1:8b")
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1")

    system = """You are an autonomous development agent.

Goal: Implement the user's request by writing code, committing, and pushing.

Stages to follow:
    - setup: Create input text file and processing script (max 10 turns, validate: test -f input.txt && test -f transform.sh)
    - transform: Apply uppercase transformation, reverse the text, and count words (max 10 turns, validate: test -f output.txt && test -f word_count.txt)
    - verify: Verify transformations are correct: check uppercase, reversal, and word count (max 10 turns, validate: bash -c )

Use bash commands to:
- Write/modify files with cat, echo, or python -c
- Run git add, git commit, git push
- Execute tests, linters, builds
- Call other LLMs via curl if needed

Each step, output a bash command in a code block. The system will execute it and show you the result. Continue until the goal is achieved.

When done, output: DONE
"""

    run(
        G=litellm_invoke(f"openai/{MODEL_ID}", base_url=BASE_URL, api_key="dummy"),
        V1=regex_parse(),
        V2=local_env(),
        emit=save_trajectory(),
        system=system,
        prompt=args.goal,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
