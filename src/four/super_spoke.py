"""Super Spoke - Most powerful spoke variant.

This combines all the best features for maximum capability:
- Tool calls for structured execution
- Auto-retry on transient failures
- Better error messages
- Context-aware execution
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .core import Err, Ok, run, retry_invoke, save_trajectory, Invoke, Validate
from .chat_model import BASH_TOOL, litellm_toolcall_invoke
from .response_model import BASH_TOOL_RESPONSE_API, http_response_invoke
from .parse import toolcall_parse
from .env import local_env, local_env_response


# ── Context-aware V2 that handles large files ───────────────────────────────


def super_env(
    timeout: int = 300,
    max_output: int = 100_000,
    exit_signal: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    stream_output: bool = True,
) -> Validate:
    """Super V2 with larger limits and streaming support.
    
    Args:
        timeout: 5 minutes default (vs 2 min for local_env)
        max_output: 100k chars (vs 10k)
        stream_output: Emit partial output during long runs
    """
    return local_env(timeout=timeout, max_output=max_output, exit_signal=exit_signal)


# ── G functions with auto-retry ─────────────────────────────────────────────


def super_invoke(
    model: str = "deep-qwen",
    base_url: str = "http://192.168.1.161:8081/v1",
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> Invoke:
    """Most powerful G: tool calls + auto-retry.
    
    Uses the deepest reasoning model (deep-qwen) with tool calling
    and exponential backoff retry on transient failures.
    """
    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        # Use tool call invoke with retry wrapper
        g = litellm_toolcall_invoke(
            model=f"openai/{model}",
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key="dummy",
        )
        return retry_invoke(g, max_attempts=15)(messages)
    
    return _invoke


def super_response_invoke(
    model: str = "deep-qwen",
    base_url: str = "http://192.168.1.161:8082/v1",
    max_output_tokens: int = 8192,
) -> Invoke:
    """Most powerful G: Responses API + auto-retry.
    
    Uses Responses API with the deepest reasoning model.
    """
    def _invoke(messages: list[dict]) -> Ok[str] | Err[str]:
        g = http_response_invoke(
            base_url=base_url,
            model=model,
            api_key="dummy",
            max_output_tokens=max_output_tokens,
        )
        return retry_invoke(g, max_attempts=15)(messages)
    
    return _invoke


# ── Super spoke variants ─────────────────────────────────────────────────────


def super_spoke_chat(
    model: str = "deep-qwen",
    base_url: str = "http://192.168.1.161:8081/v1",
    max_tokens: int = 8192,
    max_steps: int = 100,
    max_format_errors: int = 5,
    output_dir: str = "trajectories",
) -> Callable[[str], Path]:
    """Super spoke via Chat Completions + regex parsing.
    
    Best for: Tasks where the model needs to generate complex bash code.
    """
    def _run(prompt: str) -> Path:
        G = super_invoke(model=model, base_url=base_url, max_tokens=max_tokens)
        V1 = toolcall_parse()  # Can parse both tool calls and markdown blocks
        V2 = super_env()
        
        return run(
            G=G, V1=V1, V2=V2,
            emit=save_trajectory(output_dir),
            system=(
                "You are a super-powered bash agent. "
                "Execute complex tasks with multiple steps. "
                "Use tool calls for each command. "
                "When done, send COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
            ),
            prompt=prompt,
            max_steps=max_steps,
            max_format_errors=max_format_errors,
        )
    return _run


def super_spoke_responses(
    model: str = "deep-qwen",
    base_url: str = "http://192.168.1.161:8082/v1",
    max_output_tokens: int = 8192,
    max_steps: int = 100,
    max_format_errors: int = 5,
    output_dir: str = "trajectories",
) -> Callable[[str], Path]:
    """Super spoke via Responses API + tool calls.
    
    Best for: Tasks requiring the most advanced reasoning (MoE models).
    """
    def _run(prompt: str) -> Path:
        G = super_response_invoke(
            model=model,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
        )
        V1 = toolcall_parse()
        V2 = super_env()
        
        return run(
            G=G, V1=V1, V2=V2,
            emit=save_trajectory(output_dir),
            system=(
                "You are a super-powered bash agent using Responses API. "
                "Execute complex tasks with multiple steps. "
                "Use tool calls for each command. "
                "When done, send COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
            ),
            prompt=prompt,
            max_steps=max_steps,
            max_format_errors=max_format_errors,
        )
    return _run


# ── Export ───────────────────────────────────────────────────────────────────


__all__ = [
    "super_env",
    "super_invoke",
    "super_response_invoke",
    "super_spoke_chat",
    "super_spoke_responses",
]
