#!/usr/bin/env python3
"""
Autonomous pipeline: durable-state (evidence-based context transfer)

Replaces LLM-based summarization with durable state transfer via git evidence.

DISTINCTION from long-running-implementation:
    - No LLM summarization (no lossy heuristic compression)
    - Context is derived from actual evidence: git log, diffs, commit messages
    - Each stage reads the git history to understand what was built
    - Evolution is preserved as DNA in the code timeline
    - Additive: copy-enhance pattern, never deletes history

DISTINCTION from context-aware-implementation:
    - Same two-model fallback for throughput
    - Adds evidence_inject callback that prepends git state before each G call
    - Agent always sees the durable record, not compressed plausibility

Core idea:
    The git timeline IS the memory. Before each step, we inject the current
    git state (recent commits, branch status, diff summary) as context.
    The agent reads what actually happened, not what an LLM thinks happened.

    This is the physical form of compaction: small batches of commits,
    gated by PR cycles, with the four agent reading the evidence to
    determine next steps.

URLs:
    .157:8080  -> fast model (short context, high throughput)
    .161:8081  -> large model (long context, heavy reasoning)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import context_aware_invoke
from four.parse import regex_parse
from four.env import local_env, pr_gate_env
from four.core import save_trajectory


def git_evidence() -> str:
    """Collect durable state from git: recent commits, branch, status, diff summary."""
    parts = []

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts.append(f"Branch: {branch}")
    except Exception:
        parts.append("Branch: (unknown)")

    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-15"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts.append(f"Recent commits:\n{log}")
    except Exception:
        parts.append("Recent commits: (none)")

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if status:
            parts.append(f"Working tree changes:\n{status}")
        else:
            parts.append("Working tree: clean")
    except Exception:
        pass

    try:
        diff_summary = subprocess.check_output(
            ["git", "diff", "--stat"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if diff_summary:
            parts.append(f"Unstaged diff:\n{diff_summary}")
    except Exception:
        pass

    try:
        staged = subprocess.check_output(
            ["git", "diff", "--cached", "--stat"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if staged:
            parts.append(f"Staged changes:\n{staged}")
    except Exception:
        pass

    return "\n\n".join(parts)


def evidence_inject(base_invoke):
    """Wrapper that prepends current git evidence to the system message before each call.

    The git timeline serves as durable memory. Instead of summarizing
    conversation history (lossy), we inject what actually happened.
    """
    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        evidence = git_evidence()
        if not evidence:
            return base_invoke(messages)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if system_msgs:
            original = system_msgs[0].get("content", "")
            enhanced = f"""{original}

--- CURRENT REPOSITORY STATE ---
{evidence}
--- END REPOSITORY STATE ---

Use this evidence to understand what has been built, what the current state is,
and what needs to be done next. The git history is your durable memory."""
            system_msgs[0] = {"role": "system", "content": enhanced}
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"--- CURRENT REPOSITORY STATE ---\n{evidence}\n--- END REPOSITORY STATE ---",
            })

        return base_invoke(system_msgs + non_system)

    return _invoke


def main():
    parser = argparse.ArgumentParser(
        description="Durable-state autonomous pipeline with evidence-based context transfer."
    )
    parser.add_argument("--goal", required=True, help="The goal to achieve")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps")
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    LARGE_MODEL = os.getenv("FIVE_LARGE_MODEL", "qwen")
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://192.168.1.157:8080/v1")
    LARGE_URL = os.getenv("FIVE_LARGE_URL", "http://192.168.1.161:8081/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "65536"))

    step_num = [0]

    base_invoke = context_aware_invoke(
        fast_model=f"openai/{MODEL_ID}",
        large_model=f"openai/{LARGE_MODEL}",
        fast_base_url=BASE_URL,
        large_base_url=LARGE_URL,
        context_limit=50_000,
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        api_key="dummy",
    )

    invoke = evidence_inject(base_invoke)

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

CORE PRINCIPLE: Additive evolution. Never delete working code. Instead, copy-enhance:
    - Read existing code, understand its intent and limitations
    - Create improved versions alongside, referencing the original
    - The git history preserves the DNA of each decision
    - Each commit is a physical gate: small batch, reviewed, durable

RULES:
- Output ONE bash command per step in a code block.
- The system executes it and shows you the result.
- If the command fails, you will see the error — fix it and try again.
- Take small incremental steps. Do NOT try to do everything in one command.
- Use git frequently: git add, git commit after each logical unit of work.
- Commit messages must describe WHAT changed and WHY.
- Continue until the goal is fully achieved.

PR GATING — semantic cutoff:
When a logical batch of work is complete (e.g., a feature implemented, tests passing,
a module finished), signal PR creation with:
    echo 'PR_GATE: <reason for this batch>'

The system will create a PR for all commits since the last gate. You decide when
a batch is semantically complete — not by commit count, but by meaning.

Stages to follow:
    - analyze: Explore the codebase, read git log, identify what exists and what's needed. Write findings to output/analysis.md
    - plan: Create a detailed implementation plan. Break into small batches. Write to output/plan.md
    - implement: Execute the plan incrementally. Each commit is a gate: write code, test, commit, move to next.
    - validate: Run tests, linters, builds. Fix issues. Ensure clean state.
    - gate: When a logical batch is complete, signal: echo 'PR_GATE: <reason>'

IMPORTANT: Work in the current directory only. NEVER use cd to change directories.
The repository state shown above is your durable memory — use it to know what's been done.

When done, output: DONE
"""

    base_v2 = local_env()
    V2 = pr_gate_env(base_v2)

    path = run(
        G=debug_g,
        V1=regex_parse(),
        V2=V2,
        emit=save_trajectory(),
        system=system,
        prompt=args.goal,
        max_steps=args.max_steps,
    )
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")


if __name__ == "__main__":
    main()
