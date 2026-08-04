"""LiteLLM model wrapper — implements G (invoke)."""

from __future__ import annotations

import json
import logging
import os
from typing import Callable

from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from .core import Err, Invoke, Ok

logger = logging.getLogger("five.model")

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


class AbortError(Exception):
    """Error that should not be retried."""
    pass


def retry_invoke(
    invoke_fn: Invoke,
    *,
    max_attempts: int | None = None,
) -> Invoke:
    """Wrap an Invoke function with tenacity retry logic.

    Retries on transient errors (rate limits, timeouts, server errors).
    Does NOT retry on abort errors (invalid API key, auth errors, etc).
    """
    if max_attempts is None:
        max_attempts = int(os.getenv("FIVE_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))

    abort_types = (AbortError,)
    retrying = Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_not_exception_type(abort_types),
    )

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        def _call() -> Ok[str] | Err[str]:
            result = invoke_fn(messages)
            if isinstance(result, Err):
                error = result.error.lower()
                # Abort on auth/permission errors
                if any(kw in error for kw in ("invalidapi", "permission", "unauthorized", "forbidden")):
                    raise AbortError(result.error)
                # Retry on transient errors
                if any(kw in error for kw in ("rate_limit", "timeout", "too many", "server error", "overloaded")):
                    raise RuntimeError(result.error)
            return result

        try:
            return retrying(_call)
        except AbortError as e:
            return Err(f"abort: {e}")
        except Exception as e:
            return Err(f"retry_exhausted: {e}")

    return _invoke


def litellm_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    tools: list[dict] | None = None,
    **model_kwargs,
) -> Invoke:
    """Return a G function that queries the LLM via litellm."""

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
            content = response.choices[0].message.content or ""
            return Ok(content)

        except Exception as e:
            return Err(f"{type(e).__name__}: {e}")

    return _invoke


def litellm_toolcall_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Invoke:
    """Return a G function using tool-calling (structured actions)."""

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
                content = response.choices[0].message.content or ""
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
