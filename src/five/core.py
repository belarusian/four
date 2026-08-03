"""Five-function algebra for agents.

invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[action]
validate : V2  -- action → Result[observation | Exit]
fix      : G'  -- (error, messages) → message | None
emit     : IO  -- (messages, outcome) → Path

The loop: G → V1 → V2 → (G' → G)* → emit
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
Parse = Callable[[str], Result[str, str]]
Validate = Callable[[str], Result[dict, str]]
Fix = Callable[[str, list[dict]], dict | None]
Emit = Callable[[list[dict], str], Path]


# ── The loop ────────────────────────────────────────────────────────────────

def run(
    G: Invoke,
    V1: Parse,
    V2: Validate,
    G_prime: Fix,
    emit: Emit,
    system: str,
    prompt: str,
    max_steps: int = 100,
) -> Path:
    """Five-function evaluator.

    G  → query LLM, get raw text
    V1 → extract bash action from text
    V2 → execute action, get observation or Exit
    G' → format error as retry message (or None to stop)
    emit → save trajectory
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    for step in range(max_steps):
        # G: invoke
        raw = G(messages)
        if isinstance(raw, Err):
            return emit(messages, f"model_error: {raw.error}")

        # V1: parse
        action = V1(raw.value)
        if isinstance(action, Err):
            fix = G_prime(action.error, messages)
            if fix:
                messages.append(fix)
                continue
            return emit(messages, f"format_error: {action.error}")

        # V2: validate / execute
        result = V2(action.value)
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
