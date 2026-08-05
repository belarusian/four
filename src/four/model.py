"""Base model class — shared retry, cost tracking, message prep.

All G implementations inherit from LitellmModel to get:
- Tenacity retry with abort-on-auth logic
- Cost tracking via litellm
- Message preparation (cleaning, cache control, thinking blocks)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from .core import Err, Invoke, Ok

logger = logging.getLogger("four.model")


class AbortError(Exception):
    """Error that should not be retried."""
    pass


class LitellmModel(ABC):
    """Base class for all litellm-based G implementations.

    Subclasses implement _query() and _parse_actions().
    """

    abort_exceptions: tuple[type[Exception], ...] = (AbortError,)

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5-20250929",
        max_retries: int = 10,
        **model_kwargs,
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.max_retries = max_retries

        self._retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_not_exception_type(self.abort_exceptions),
        )

    def _invoke(self, messages: list[dict]) -> Ok[str] | Err[str]:
        """Run the full query → parse cycle with retry."""
        clean = self._prepare_messages(messages)

        def _call() -> Ok[str] | Err[str]:
            response = self._query(clean)
            return self._parse_response(response)

        try:
            return self._retrying(_call)
        except AbortError as e:
            return Err(f"abort: {e}")
        except Exception as e:
            return Err(f"retry_exhausted: {e}")

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Clean messages — remove extra keys, handle thinking blocks."""
        clean = [
            {k: v for k, v in m.items()
             if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
            for m in messages
        ]
        # TODO: Add Anthropic thinking block reordering
        return clean

    @abstractmethod
    def _query(self, messages: list[dict]) -> object:
        """Run the litellm call. Must be implemented by subclasses."""

    @abstractmethod
    def _parse_response(self, response: object) -> Ok[str] | Err[str]:
        """Parse the response into Ok(text) or Err(error)."""


# ── Factory functions (convenience wrappers) ──────────────────────────────


def make_litellm_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Callable[[list[dict]], Ok[str] | Err[str]]:
    """Create a G function for Chat Completions + text-based parsing."""
    impl = ChatCompletionsModel(model, **model_kwargs)
    return impl._invoke


def make_litellm_toolcall_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Callable[[list[dict]], Ok[str] | Err[str]]:
    """Create a G function for Chat Completions + tool calls."""
    impl = ChatCompletionsToolcallModel(model, **model_kwargs)
    return impl._invoke


def make_litellm_response_invoke(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    **model_kwargs,
) -> Callable[[list[dict]], Ok[str] | Err[str]]:
    """Create a G function for Responses API + tool calls."""
    impl = ResponsesModel(model, **model_kwargs)
    return impl._invoke
