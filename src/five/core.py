"""Five-function algebra for agents.

invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[list[action]]
validate : V2  -- action → Result[observation | Exit]
emit     : IO  -- (messages, outcome) → Path

The loop: (G → V1 → V2*)* → emit

Format errors are appended as user messages — no inner retry loop.
Consecutive format errors are tracked and abort after N failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, TypeAlias, Union

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Ok(Generic[T]):
    value: T


@dataclass
class Err(Generic[E]):
    error: E


Result: TypeAlias = Union[Ok[T], Err[E]]


# ── Signatures ──────────────────────────────────────────────────────────────

Invoke = Callable[[list[dict]], Result[str, str]]
Parse = Callable[[str], Result[list[str], str]]
Validate = Callable[[str], Result[dict, str]]
Emit = Callable[[list[dict], str], Path]


# ── The loop ────────────────────────────────────────────────────────────────

def run(
    G: Invoke,
    V1: Parse,
    V2: Validate,
    emit: Emit,
    system: str,
    prompt: str,
    max_steps: int = 100,
    max_format_errors: int = 3,
) -> Path:
    """Five-function evaluator.

    Loop: (G → V1 → V2*)*, repeat until V2 exits or max_steps.

    G  → query LLM, get raw text
    V1 → extract bash actions from text (returns list)
    V2 → execute each action, get observation or Exit
    emit → save trajectory

    Format errors are appended as user messages and the loop continues.
    Consecutive format errors beyond max_format_errors abort the loop.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    consecutive_format_errors = 0

    for step in range(max_steps):
        # G: invoke
        raw = G(messages)
        if isinstance(raw, Err):
            return emit(messages, f"model_error: {raw.error}")

        # V1: parse
        actions = V1(raw.value)
        if isinstance(actions, Err):
            consecutive_format_errors += 1
            if 0 < max_format_errors <= consecutive_format_errors:
                return emit(messages, f"repeated_format_error: {actions.error}")
            messages.append({
                "role": "user",
                "content": f"Format error: {actions.error}. Please respond with exactly one bash command in the expected format.",
            })
            continue

        consecutive_format_errors = 0

        # V2: validate / execute each action
        for action in actions.value:
            result = V2(action)
            if isinstance(result, Err):
                return emit(messages, result.error)
            messages.append(result.value)

    return emit(messages, "max_steps_reached")


# ── Trajectory I/O ─────────────────────────────────────────────────────────


def save_trajectory(
    output_dir: Path | str = "trajectories",
) -> Emit:
    """Return an emit function that saves trajectories as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _emit(messages: list[dict], outcome: str) -> Path:
        idx = len(list(out.glob("*.json")))
        path = out / f"trajectory_{idx:04d}.json"
        path.write_text(
            json.dumps({"outcome": outcome, "messages": messages}, indent=2)
        )
        return path

    return _emit
