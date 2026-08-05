"""Shared retry logic for all G functions."""

from __future__ import annotations

import logging
import os

from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from .core import Err, Invoke, Ok

logger = logging.getLogger("four.model")


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
