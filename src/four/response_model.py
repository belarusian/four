"""Responses API G functions — direct HTTP /v1/responses."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .core import Err, Invoke, Ok

# ── Responses API tool definition (flat structure, no nested "function") ───

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


def http_response_invoke(
    base_url: str = "http://192.168.1.157:8080/v1",
    model: str = "fast-qwen",
    api_key: str = "sk-not-needed",
    max_output_tokens: int = 4096,
    timeout: int = 120,
) -> Invoke:
    """G via direct HTTP to /v1/responses.

    Converts message history to Responses API format before each call.
    Returns JSON array of {tool_call_id, name, arguments}.
    V1 should use toolcall_response_parse().
    """
    import litellm

    def _to_input_format(messages: list[dict]) -> list[dict]:
        """Convert chat messages to Responses API input items."""
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
        input_items = _to_input_format(messages)

        payload = {
            "model": model,
            "input": input_items,
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
                tool_calls = [i for i in output if i.get("type") == "function_call"]

                if not tool_calls:
                    # No tool calls — return text content
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


def litellm_response_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Invoke:
    """G via litellm.responses() — Responses API with tool calls."""
    import litellm

    def _to_input_format(messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool":
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
                result.append(msg)
        return result

    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        input_items = _to_input_format(messages)

        try:
            response = litellm.responses(
                model=model,
                input=input_items,
                tools=[BASH_TOOL_RESPONSE_API],
                **model_kwargs,
            )

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
