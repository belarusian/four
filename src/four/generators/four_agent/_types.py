"""Four-agent generator types.

For generating four-style functional algebra agents:
- G: messages -> Result[raw]
- V1: raw -> Result[actions]
- V2: action -> Result[observation | Exit]
- emit: (messages, outcome) -> Path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from four.generators._types import Err, Ok, Result


@dataclass(frozen=True)
class FourAgentSpec:
    """Complete four-style agent runner specification.

    A Python script that implements a four-function algebra agent runner.

    Response format -- Python constructor with source as triple-quoted string:

    FourAgentSpec(
        title="Bash Agent",
        description="Runs a four-style agent with chat variant",
        variant="chat",
        system_prompt="You are a bash agent...",
        source=\"\"\"#!/usr/bin/env python3
import ...
def main():
    ...
\"\"\",
    )
    """
    title: str
    description: str
    variant: str  # "chat", "toolcall", or "responses"
    system_prompt: str
    source: str   # The complete Python source code for the agent


@dataclass(frozen=True)
class Edit:
    """A text edit: find old, replace with new."""
    old: str
    new: str


@dataclass(frozen=True)
class AgentPatch:
    """A patch to apply to the agent source code."""
    edits: tuple[Edit, ...] = ()


def validate_spec_instance(spec: FourAgentSpec) -> Result[FourAgentSpec, str]:
    """Validate a FourAgentSpec instance."""
    errors: list[str] = []

    if not spec.title:
        errors.append("title must be non-empty")

    if not spec.description:
        errors.append("description must be non-empty")

    valid_variants = {"chat", "toolcall", "responses"}
    if spec.variant not in valid_variants:
        errors.append(f"variant must be one of {valid_variants}")

    if not spec.system_prompt and spec.variant != "toolcall":
        errors.append("system_prompt must be non-empty for non-toolcall variants")

    if not spec.source:
        errors.append("source must be non-empty")

    if errors:
        return Err("; ".join(errors))

    return Ok(spec)


def apply_patch(
    spec: FourAgentSpec,
    patch: AgentPatch,
) -> Result[FourAgentSpec, str]:
    """Apply a Patch to a FourAgentSpec's source code."""
    source = spec.source
    errors: list[str] = []

    for edit in patch.edits:
        if edit.old not in source:
            errors.append(f"Edit 'old' not found in source: {edit.old[:60]}...")
        else:
            source = source.replace(edit.old, edit.new, 1)

    if errors:
        return Err("; ".join(errors))

    return Ok(FourAgentSpec(
        title=spec.title,
        description=spec.description,
        variant=spec.variant,
        system_prompt=spec.system_prompt,
        source=source,
    ))
