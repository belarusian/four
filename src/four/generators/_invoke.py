"""
Model invocation and prompt building for generators.

Standalone version: uses OpenAI-compatible endpoints (llama.cpp, vLLM, etc.)
via urllib. No dependency on Compass's Provider abstraction.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from ._types import AskFn, Err, GenerationContext, Ok, Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone OpenAI-compatible provider
# ---------------------------------------------------------------------------


class SimpleProvider:
    """Minimal provider for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: int = 120,
        api_key: str = "sk-local",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    @property
    def name(self) -> str:
        return f"openai/{self.base_url}"

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = -1,
        temperature: float = 0.3,
    ) -> dict:
        """Send a chat completion request."""
        payload = {
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choice = body.get("choices", [{}])[0]
                message = choice.get("message", {})
                return {
                    "text": message.get("content", "").strip(),
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection failed: {e.reason}") from e


def _provider_ask(provider: SimpleProvider) -> AskFn:
    """Wrap a SimpleProvider as an AskFn."""
    max_tokens = -1

    def ask(system: str, user: str) -> Result:
        sys_chars = len(system)
        usr_chars = len(user)
        t0 = time.monotonic()
        try:
            resp = provider.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            elapsed = time.monotonic() - t0
            text = resp["text"]
            logger.info(
                "[%s] %.1fs | prompt %d+%d chars | response %d chars",
                provider.name, elapsed, sys_chars, usr_chars, len(text),
            )
            return Ok(text) if text else Err("Empty response from provider")
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[%s] %.1fs | prompt %d+%d chars | ERROR: %s",
                provider.name, elapsed, sys_chars, usr_chars, e,
            )
            return Err(f"Provider error: {e}")
    return ask


def resolve_ask_fn(base_url: str = "", ask_fn: AskFn | None = None) -> AskFn:
    """Build the model invocation function.

    Resolution: ask_fn > SimpleProvider(base_url).

    When base_url is not set, falls back to env vars:
        LLM_BASE_URL or LLAMACPP_HOST
    """
    if ask_fn is not None:
        return ask_fn

    url = base_url or __import__("os").environ.get("LLM_BASE_URL") or __import__("os").environ.get("LLAMACPP_HOST", "http://localhost:8080/v1")
    provider = SimpleProvider(url)
    logger.info("Generator using provider: %s", provider.name)
    return _provider_ask(provider)


# ---------------------------------------------------------------------------
# Prompt building (pure)
# ---------------------------------------------------------------------------


def build_system_prompt(
    ctx: GenerationContext,
    types_source: str,
    *,
    role: str = "You are an expert at generating structured artifacts.",
    contract_preamble: str = (
        "Your output must conform to the Spec type below."
    ),
) -> str:
    """Assemble the system prompt from context."""
    sections = [
        role,
        "",
        "## Output Contract (Python types)",
        "",
        contract_preamble,
        "",
        types_source,
    ]

    for ds in ctx.domain_context:
        if ds.content and not ds.content.startswith("No "):
            sections.extend(["", f"## {ds.heading}", "", ds.content])

    return "\n".join(sections)


def build_user_message(
    ctx: GenerationContext,
    *,
    focus: str | None = None,
    suffix_lines: tuple[str, ...] = (
        "Respond with the Spec type defined in the Output Contract.",
        "See its docstring for the response format.",
        "No markdown fencing, no explanation.",
    ),
) -> str:
    """Assemble the user message from context."""
    primary = (
        ctx.user_prompt if ctx.user_prompt is not None else
        ctx.default_task if ctx.default_task is not None else
        "Generate an artifact."
    )
    parts = [primary]

    parts.extend(["", *suffix_lines])

    if ctx.available_packages:
        parts.extend(["", f"Available packages: {ctx.available_packages}"])

    if focus:
        parts.extend(["", f"FOCUS AREA: Emphasise '{focus}' patterns."])

    if ctx.feedback:
        parts.extend(["", "Your previous attempt had errors:", ""])
        for fb in ctx.feedback:
            parts.append(f"  {fb}")
        parts.extend(["", "Please fix these issues in your next attempt."])

    return "\n".join(parts)
