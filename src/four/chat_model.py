"""Chat Completions API G functions — litellm.completion()."""

from __future__ import annotations

import json

from .core import Err, Invoke, Ok

# ── Chat Completions tool definition (nested "function" key) ───────────────

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the shell.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}


def litellm_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    tools: list[dict] | None = None,
    **model_kwargs,
) -> Invoke:
    """G via litellm.completion() — plain text response."""

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        import litellm

        clean = [
            {k: v for k, v in m.items()
             if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
            for m in messages
        ]

        try:
            kwargs = {"model": model, "messages": clean, **model_kwargs}
            if tools:
                kwargs["tools"] = tools

            response = litellm.completion(**kwargs)
            msg = response.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", "") or ""
            return Ok(content)

        except Exception as e:
            return Err(f"{type(e).__name__}: {e}")

    return _invoke


def litellm_toolcall_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Invoke:
    """G via litellm.completion() with tool calls.

    Returns JSON array of {tool_call_id, name, arguments}.
    V1 should use toolcall_parse().
    """

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        import litellm

        clean = [
            {k: v for k, v in m.items()
             if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
            for m in messages
        ]

        try:
            response = litellm.completion(
                model=model,
                messages=clean,
                tools=[BASH_TOOL],
                **model_kwargs,
            )

            tool_calls = response.choices[0].message.tool_calls or []
            if not tool_calls:
                msg = response.choices[0].message
                content = msg.content or getattr(msg, "reasoning_content", "") or ""
                return Ok(content)

            actions = []
            for tc in tool_calls:
                func = tc.function
                actions.append({
                    "tool_call_id": tc.id,
                    "name": func.name,
                    "arguments": func.arguments,
                })

            return Ok(json.dumps(actions))

        except Exception as e:
            return Err(f"{type(e).__name__}: {e}")

    return _invoke
