"""Action parsers — implement V1 (parse)."""

from __future__ import annotations

import json
import re

from .core import Err, Ok, Parse


def regex_parse(
    pattern: str | None = None,
    error_template: str = "Found {count} actions. Expected at least 1.",
) -> Parse:
    """Parse bash commands from markdown code blocks.

    Accepts ```mswea_bash_command, ```bash, or ```sh blocks (in that priority order).
    Returns all matched commands as a list.
    """
    if pattern is None:
        # Try multiple block types in order of specificity
        # Handles both ```bash\ncommand\n``` and ```bash command```
        patterns = [
            r"```mswea_bash_command\s*(?:\n|\s)(.*?)(?:\n|\s)```",
            r"```bash\s+(.*?)```",           # single-line: ```bash command```
            r"```bash\s*\n(.*?)\n```",       # multi-line: ```bash\ncommand\n```
            r"```sh\s+(.*?)```",
            r"```sh\s*\n(.*?)\n```",
            r"```\s+(.*?)```",
            r"```\s*\n(.*?)\n```",           # fallback: any code block
        ]
    else:
        patterns = [pattern]

    def _parse(raw: str) -> Ok[list[str]] | Err[str]:
        for p in patterns:
            matches = re.findall(p, raw, re.DOTALL)
            if matches:
                return Ok([m.strip() for m in matches])
        return Err(error_template.format(count=0))

    return _parse


def toolcall_parse() -> Parse:
    """Parse tool-calling JSON into bash commands.

    Expects JSON array of {tool_call_id, name, arguments} where
    arguments is a JSON string with a 'command' field.
    Returns all bash commands as a list.
    """

    def _parse(raw: str) -> Ok[list[str]] | Err[str]:
        try:
            actions = json.loads(raw)
        except json.JSONDecodeError:
            # Not tool-call JSON — treat as plain text
            return Ok([raw])

        if not isinstance(actions, list) or not actions:
            return Err("No tool calls found")

        commands = []
        for action in actions:
            if action.get("name") == "bash":
                try:
                    args = json.loads(action["arguments"])
                    commands.append(args["command"])
                except (json.JSONDecodeError, KeyError) as e:
                    return Err(f"Invalid bash tool call: {e}")

        if not commands:
            return Err("No bash tool call found")

        return Ok(commands)

    return _parse
