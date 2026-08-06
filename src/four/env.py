"""Local shell environment — implements V2 (validate/execute).

Supports both Chat Completions and Responses API observation formats.
"""

from __future__ import annotations

import subprocess
import sys

from .core import Err, Ok, Validate


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
