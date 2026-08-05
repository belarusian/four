"""Responses API G — inherits from LitellmModel.

Like mini-swe-agent's LitellmResponseModel.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .core import Err, Ok
from .model import LitellmModel

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


class _ResponsesBase(LitellmModel):
    """Base for Responses API implementations."""

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into output items for stateless API calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result


class _ResponsesLitellm(_ResponsesBase):
    """Responses API via litellm.responses()."""

    def _query(self, messages: list[dict]) -> object:
        import litellm
        return litellm.responses(
            model=self.model,
            input=messages,
            tools=[BASH_TOOL_RESPONSE_API],
            **self.model_kwargs,
        )

    def _parse_response(self, response) -> Ok[str] | Err[str]:
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


class _ResponsesHTTP(_ResponsesBase):
    """Responses API via direct HTTP to /v1/responses (bypasses litellm)."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 max_output_tokens: int, timeout: int, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.model = model
        self.model_kwargs = kwargs
        self._retrying = None  # HTTP impl doesn't use retry

    def _query(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self.max_output_tokens,
            "tools": [BASH_TOOL_RESPONSE_API],
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/responses"
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse_response(self, response: dict) -> Ok[str] | Err[str]:
        output = body.get("output", [])
        tool_calls = [i for i in output if i.get("type") == "function_call"]

        if not tool_calls:
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

    def _invoke(self, messages: list[dict]) -> Ok[str] | Err[str]:
        """Direct HTTP has no retry — just call and parse."""
        clean = self._prepare_messages(messages)
        try:
            response = self._query(clean)
            return self._parse_response(response)  # type: ignore[arg-type]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            return Err(f"HTTP {e.code}: {err_body[:500]}")
        except urllib.error.URLError as e:
            return Err(f"Connection failed: {e.reason}")
        except Exception as e:
            return Err(f"{type(e).__name__}: {e}")


# ── Public factory functions ───────────────────────────────────────────────


def litellm_response_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
):
    """G via litellm.responses() — Responses API with tool calls."""
    impl = _ResponsesLitellm(model, **model_kwargs)
    return impl._invoke


def http_response_invoke(
    base_url: str = "http://192.168.1.157:8080/v1",
    model: str = "fast-qwen",
    api_key: str = "sk-not-needed",
    max_output_tokens: int = 4096,
    timeout: int = 120,
):
    """G via direct HTTP to /v1/responses — bypasses litellm's auth check."""
    impl = _ResponsesHTTP(base_url, model, api_key, max_output_tokens, timeout)
    return impl._invoke
