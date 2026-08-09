"""Local shell environment — implements V2 (validate/execute).

Supports both Chat Completions and Responses API observation formats.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .core import Err, Ok, Validate


def pr_gate_env(
    base_v2: Validate,
    repo_name: str = "personal-index",
    owner: str = "belarusian",
) -> Validate:
    """V2 wrapper that intercepts PR_GATE signals from the agent.

    When the agent outputs a command containing 'PR_GATE:', this wrapper
    creates a PR for the current batch of commits. The agent decides
    semantically when a batch is complete.

    Usage in system prompt:
        "When a logical batch of work is complete, signal PR creation with:
         echo 'PR_GATE: <reason for this batch>'"

    Args:
        base_v2: The underlying V2 to wrap (e.g., local_env())
        repo_name: GitHub repo name
        owner: GitHub owner

    Returns:
        V2 that intercepts PR_GATE signals and creates PRs.
    """
    def _validate(command: str) -> Ok[dict] | Err[str]:
        # Check for PR_GATE signal
        if "PR_GATE:" in command:
            reason = command.split("PR_GATE:", 1)[1].strip().strip("'\"")
            if not reason:
                reason = "autonomous batch complete"

            try:
                # Count commits since last gate
                gate_file = Path(".pr_gate_state")
                if gate_file.exists():
                    old_hash = gate_file.read_text().strip().split("\n")[0]
                    count = subprocess.check_output(
                        ["git", "rev-list", "--count", f"{old_hash}..HEAD"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()
                else:
                    count = "0"

                pr_num = 0
                if gate_file.exists():
                    parts = gate_file.read_text().strip().split("\n")
                    if len(parts) > 1:
                        pr_num = int(parts[1])

                new_pr = pr_num + 1
                branch = f"autonomous/pr-{new_pr}"

                # Create branch and push
                subprocess.run(
                    ["git", "checkout", "-b", branch],
                    capture_output=True,
                )
                push_result = subprocess.run(
                    ["git", "push", "origin", branch],
                    capture_output=True, text=True,
                )

                if push_result.returncode != 0:
                    # Branch may already exist, force push
                    subprocess.run(
                        ["git", "push", "origin", branch, "--force-with-lease"],
                        capture_output=True,
                    )

                # Build PR body
                if gate_file.exists():
                    old_hash = gate_file.read_text().strip().split("\n")[0]
                    logs = subprocess.check_output(
                        ["git", "log", "--oneline", f"{old_hash}..HEAD"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()
                else:
                    logs = subprocess.check_output(
                        ["git", "log", "--oneline", "-10"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()

                body = f"Reason: {reason}\n\nCommits in this batch:\n```\n{logs}\n```"

                # Create PR
                pr_result = subprocess.run(
                    [
                        "gh", "pr", "create",
                        "--base", "main",
                        "--head", branch,
                        "--title", f"Autonomous batch {new_pr}: {reason}",
                        "--body", body,
                    ],
                    capture_output=True, text=True,
                )

                # Return to main
                subprocess.run(
                    ["git", "checkout", "main"],
                    capture_output=True,
                )

                # Update gate state
                new_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                gate_file.write_text(f"{new_hash}\n{new_pr}")

                if pr_result.returncode == 0:
                    output = f"PR #{new_pr} created: {pr_result.stdout.strip()}"
                else:
                    output = f"PR create result: {pr_result.stderr.strip()}"

                return Ok({
                    "role": "tool",
                    "content": f"<returncode>0</returncode>\n<output>\nPR_GATE triggered ({count} commits):\n{output}\n</output>",
                })

            except Exception as e:
                return Ok({
                    "role": "tool",
                    "content": f"<returncode>1</returncode>\n<output>\nPR_GATE error: {e}\n</output>",
                })

        return base_v2(command)

    return _validate


def local_env(
    timeout: int = 120,
    max_output: int = 10_000,
    exit_signal: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
) -> Validate:
    """Return a V2 function that executes bash commands locally.

    Returns observations in Chat Completions format: {"role": "tool", "content": ...}
    """

    def _validate(command: str) -> Ok[dict] | Err[str]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Err("timeout")
        except Exception as e:
            return Err(f"execution_error: {e}")

        output = result.stdout + result.stderr
        lines = output.splitlines()

        # Check for exit signal
        if lines and lines[0].strip() == exit_signal and result.returncode == 0:
            return Err("exit:task_complete")

        # Truncate long output
        if len(output) > max_output:
            output = (
                output[: max_output // 2]
                + f"\n... [{len(output) - max_output} chars elided] ...\n"
                + output[-max_output // 2 :]
            )

        observation = {
            "role": "tool",
            "content": (
                f"<returncode>{result.returncode}</returncode>\n"
                f"<output>\n{output}\n</output>"
            ),
        }

        return Ok(observation)

    return _validate


def local_env_response(
    timeout: int = 120,
    max_output: int = 10_000,
    exit_signal: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
) -> Validate:
    """Return a V2 function that executes bash commands locally.

    Returns observations in Responses API format:
    {"type": "function_call_output", "call_id": ..., "output": ...}

    The call_id is passed through the action dict (embedded as #call_id:).
    """

    _last_call_id = "call_unknown"

    def _validate(action: str) -> Ok[dict] | Err[str]:
        nonlocal _last_call_id

        # Extract call_id if embedded in action
        command = action
        if action.startswith("#call_id:"):
            parts = action.split("\n", 1)
            _last_call_id = parts[0].split(":", 1)[1]
            command = parts[1] if len(parts) > 1 else ""

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Err("timeout")
        except Exception as e:
            return Err(f"execution_error: {e}")

        output = result.stdout + result.stderr
        lines = output.splitlines()

        if lines and lines[0].strip() == exit_signal and result.returncode == 0:
            return Err("exit:task_complete")

        if len(output) > max_output:
            output = (
                output[: max_output // 2]
                + f"\n... [{len(output) - max_output} chars elided] ...\n"
                + output[-max_output // 2 :]
            )

        # For Responses API, return as function_call_output
        content = (
            f"<returncode>{result.returncode}</returncode>\n"
            f"<output>\n{output}\n</output>"
        )

        observation = {
            "type": "function_call_output",
            "call_id": _last_call_id,
            "output": content,
        }

        return Ok(observation)

    return _validate
