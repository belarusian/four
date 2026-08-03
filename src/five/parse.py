"""Action parsers — implement V1 (parse)."""

from __future__ import annotations

import json
import re

from .core import Err, Ok, Parse


def regex_parse(
    pattern: str | None = None,
    error_template: str = "Found {count} actions. Expected exactly 1.",
) -> Parse:
    """Parse a single bash command from markdown code blocks.

    Accepts ```mswea_bash_command, ```bash, or ```sh blocks (in that priority order).
    """
    if pattern is None:
        # Try multiple block types in order of specificity
        patterns = [
            r"```mswea_bash_command\s*\n(.*?)\n```",
            r"```bash\s*\n(.*?)\n```",
            r"```sh\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",  # fallback: any code block
        ]
    else:
        patterns = [pattern]

    def _parse(raw: str) -> Ok[str] | Err[str]:
        for p in patterns:
            matches = re.findall(p, raw, re.DOTALL)
            if matches:
                # Return the first match from the most specific pattern that worked
                return Ok(matches[0].strip())
        return Err(error_template.format(count=0))

    return _parse


def toolcall_parse() -> Parse:
    """Parse tool-calling JSON into a bash command.

    Expects JSON array of {tool_call_id, name, arguments} where
    arguments is a JSON string with a 'command' field.
    """

    def _parse(raw: str) -> Ok[str] | Err[str]:
        try:
            actions = json.loads(raw)
        except json.JSONDecodeError:
            # Not tool-call JSON — treat as plain text (might have regex pattern)
            return Ok(raw)

        if not isinstance(actions, list) or not actions:
            return Err("No tool calls found")

        # Take the first bash tool call
        for action in actions:
            if action.get("name") == "bash":
                try:
                    args = json.loads(action["arguments"])
                    return Ok(args["command"])
                except (json.JSONDecodeError, KeyError) as e:
                    return Err(f"Invalid bash tool call: {e}")

        return Err("No bash tool call found")

    return _parse
