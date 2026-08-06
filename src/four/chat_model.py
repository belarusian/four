"""Chat Completions API G — inherits from LitellmModel.

Two variants:
1. Text-based (regex parsing) — like mini-swe-agent's LitellmTextbasedModel
2. Tool calls — like mini-swe-agent's LitellmModel
"""

from __future__ import annotations

import json

from .core import Err, Ok
from .model import LitellmModel

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


class _ChatCompletionsBase(LitellmModel):
    """Base for Chat Completions API implementations."""
    
    BASH_TOOL = BASH_TOOL

    def _query(self, messages: list[dict]) -> object:
        import litellm
        kwargs = {"model": self.model, "messages": messages, **self.model_kwargs}
        if hasattr(self, "BASH_TOOL"):
            kwargs["tools"] = [self.BASH_TOOL]
        return litellm.completion(**kwargs)


class _ChatCompletionsText(LitellmModel):
    """Chat Completions + text-based parsing (regex) — NO tools."""

    def _query(self, messages: list[dict]) -> object:
        import litellm
        kwargs = {"model": self.model, "messages": messages, **self.model_kwargs}
        # Explicitly do NOT pass tools - we want plain text output with markdown code blocks
        return litellm.completion(**kwargs)

    def _parse_response(self, response) -> Ok[str] | Err[str]:
        msg = response.choices[0].message
        
        # If tool calls are present, return them as JSON
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            actions = []
            for tc in tool_calls:
                func = tc.function
                actions.append({
                    "tool_call_id": tc.id,
                    "name": func.name,
                    "arguments": func.arguments,
                })
            return Ok(json.dumps(actions))
        
        # Otherwise return text content or reasoning
        content = msg.content or getattr(msg, "reasoning_content", "") or ""
        return Ok(content)


class _ChatCompletionsToolcall(_ChatCompletionsBase):
    """Chat Completions + tool calls."""

    def _parse_response(self, response) -> Ok[str] | Err[str]:
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


# ── Public factory functions ───────────────────────────────────────────────


def litellm_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
):
    """G via litellm.completion() — plain text response."""
    impl = _ChatCompletionsText(model, **model_kwargs)
    return impl._invoke


def litellm_toolcall_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
):
    """G via litellm.completion() with tool calls.

    Returns JSON array of {tool_call_id, name, arguments}.
    V1 should use toolcall_parse().
    """
    impl = _ChatCompletionsToolcall(model, **model_kwargs)
    return impl._invoke
