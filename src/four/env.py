"""Local shell environment — implements V2 (validate/execute)."""

from __future__ import annotations

import subprocess
import sys

from .core import Err, Ok, Validate


def local_env(
    timeout: int = 120,
    max_output: int = 10_000,
    exit_signal: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
) -> Validate:
    """Return a V2 function that executes bash commands locally."""

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
