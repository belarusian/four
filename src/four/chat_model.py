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


def context_aware_invoke(
    fast_model: str,
    large_model: str,
    *,
    fast_base_url: str = "",
    large_base_url: str = "",
    context_limit: int = 50_000,
    **model_kwargs,
):
    """G that switches to large-context model when conversation grows too big.

    Estimates token count from message text (~4 chars/token). When context
    exceeds context_limit, switches to the large-context model. When even
    the large model can't handle it, compresses history by keeping system
    + last 8 messages.

    Args:
        fast_model: Model ID for normal operations (e.g., "openai/fast-qwen")
        large_model: Model ID for large context (e.g., "openai/qwen")
        fast_base_url: API URL for fast model
        large_base_url: API URL for large model
        context_limit: Token threshold to switch models (default 50k)
        **model_kwargs: Additional kwargs passed to litellm
    """
    import logging
    logger = logging.getLogger("four.model")

    fast_impl = _ChatCompletionsText(fast_model, base_url=fast_base_url, **model_kwargs)
    large_impl = _ChatCompletionsText(large_model, base_url=large_base_url, **model_kwargs)

    def _estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total = sum(len(str(m.get("content", ""))) for m in messages)
        return total // 4

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        estimated = _estimate_tokens(messages)

        if estimated > 200_000:
            logger.warning("Context %d tokens too large, compressing history", estimated)
            system = [m for m in messages if m.get("role") == "system"]
            recent = messages[-8:]
            return large_impl._invoke(system + recent)

        if estimated > context_limit:
            logger.info("Context %d tokens > %d, switching to large model", estimated, context_limit)
            return large_impl._invoke(messages)

        return fast_impl._invoke(messages)

    return _invoke
