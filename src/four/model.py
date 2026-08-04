"""LiteLLM model wrapper — implements G (invoke)."""

from __future__ import annotations

import json
import logging
import os
from typing import Callable

from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from .core import Err, Invoke, Ok

logger = logging.getLogger("four.model")

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

# ── Responses API tool definition (flat structure, no nested "function") ────

BASH_TOOL_RESPONSE_API = {
    "type": "function",
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


# ── Responses API G ────────────────────────────────────────────────────────


def litellm_response_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Invoke:
    """G via litellm.responses() — Responses API with tool calls.

    Uses flat function definitions (no nested "function" key).
    Returns JSON array of {call_id, name, arguments}.
    V1 should use toolcall_response_parse().
    """

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        import litellm

        # Flatten response objects into output items for stateless API calls
        clean = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    clean.append({k: v for k, v in item.items() if k != "extra"})
            else:
                clean.append({k: v for k, v in msg.items() if k != "extra"})

        try:
            response = litellm.responses(
                model=model,
                input=clean,
                tools=[BASH_TOOL_RESPONSE_API],
                **model_kwargs,
            )

            # Responses API returns output as a list of items
            output = getattr(response, "output", []) or []
            tool_calls = []
            for item in output:
                item_type = (item.get("type") if isinstance(item, dict)
                             else getattr(item, "type", None))
                if item_type == "function_call":
                    td = item.model_dump() if hasattr(item, "model_dump") else \
                         dict(item) if not isinstance(item, dict) else item
                    tool_calls.append(td)

            if not tool_calls:
                # No tool calls — return text content as plain string
                text_items = [i for i in output if i.get("type") == "message"
                              and any(c.get("type") == "output_text"
                                      for c in i.get("content", []))]
                if text_items:
                    text = " ".join(
                        c.get("text", "")
                        for item in text_items
                        for c in item.get("content", [])
                        if isinstance(c, dict) and c.get("type") == "output_text"
                    )
                    return Ok(text)
                return Ok("")

            actions = []
            for tc in tool_calls:
                actions.append({
                    "tool_call_id": tc.get("call_id") or tc.get("id"),
                    "name": tc.get("name"),
                    "arguments": tc.get("arguments", "{}"),
                })

            return Ok(json.dumps(actions))

        except Exception as e:
            return Err(f"{type(e).__name__}: {e}")

    return _invoke


# ── Direct HTTP Responses API G (bypasses litellm auth check) ──────────────


def http_response_invoke(
    base_url: str = "http://192.168.1.157:8080/v1",
    model: str = "fast-qwen",
    api_key: str = "sk-not-needed",
    max_output_tokens: int = 4096,
    timeout: int = 120,
) -> Invoke:
    """G via direct HTTP to /v1/responses — bypasses litellm's auth check.

    Uses flat function definitions (no nested "function" key).
    Returns JSON array of {call_id, name, arguments}.
    V1 should use toolcall_response_parse().

    Converts message history to Responses API format before each call.
    """
    import urllib.request
    import urllib.error

    def _to_response_format(messages: list[dict]) -> list[dict]:
        """Convert chat completions messages to Responses API format."""
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool":
                # Convert tool responses to function_call_output
                call_id = msg.get("tool_call_id", "call_unknown")
                result.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content,
                })
            elif role in ("user", "assistant", "system"):
                result.append({
                    "type": "message",
                    "role": role,
                    "content": content,
                })
            else:
                # Already in Responses API format
                result.append(msg)
        return result

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        clean = _to_response_format(messages)

        payload = {
            "model": model,
            "input": clean,
            "max_output_tokens": max_output_tokens,
            "tools": [BASH_TOOL_RESPONSE_API],
        }

        data = json.dumps(payload).encode("utf-8")
        url = f"{base_url}/responses"

        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

                output = body.get("output", [])
                tool_calls = []
                for item in output:
                    if item.get("type") == "function_call":
                        tool_calls.append(item)

                if not tool_calls:
                    # No tool calls — return text content as plain string
                    text_items = [i for i in output if i.get("type") == "message"]
                    if text_items:
                        text = " ".join(
                            c.get("text", "")
                            for item in text_items
                            for c in item.get("content", [])
                            if isinstance(c, dict) and c.get("type") == "output_text"
                        )
                        return Ok(text)
                    return Ok("")

                actions = []
                for tc in tool_calls:
                    actions.append({
                        "tool_call_id": tc.get("call_id") or tc.get("id"),
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments", "{}"),
                    })

                return Ok(json.dumps(actions))

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            return Err(f"HTTP {e.code}: {err_body[:500]}")
        except urllib.error.URLError as e:
            return Err(f"Connection failed: {e.reason}")
        except Exception as e:
            return Err(f"{type(e).__name__}: {e}")

    return _invoke
