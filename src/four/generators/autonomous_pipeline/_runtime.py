"""Runtime functions for autonomous pipeline generation.

G_auto: invoke model, return AutonomousPipelineSpec
V_auto: validate spec instance
G'_auto: patch-based ouroboros
IO: emit .py spoke file
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from four.generators._types import (
    AskFn,
    DomainSection,
    Err,
    GenerationContext,
    Ok,
    Result,
)

from four.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)

from ._types import AutonomousPipelineSpec, AutonomousStage

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


def invoke_model(
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return an AutonomousPipelineSpec."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert at generating autonomous pipeline spokes for the four framework. "
            "Generate a Python script that implements a long-running autonomous development loop."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing AutonomousPipelineSpec.\n"
            "See the docstring for the exact response format."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write an AutonomousPipelineSpec(...) expression.",
            "Use dict syntax for stages: stages={\"stage1\": AutonomousStage(name=\"...\", description=\"...\", max_turns=N)}",
            "No markdown fencing.",
        ),
    )

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    try:
        spec = _parse_response(raw_text)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


def _parse_response(raw_text: str) -> AutonomousPipelineSpec:
    """Parse the model's Python constructor response into a spec."""
    start_marker = "AutonomousPipelineSpec("
    start_idx = raw_text.find(start_marker)

    if start_idx < 0:
        raise ValueError("AutonomousPipelineSpec constructor not found in response")

    # Extract the constructor body by balancing parens
    depth = 0
    end_idx = start_idx
    for i in range(start_idx, len(raw_text)):
        ch = raw_text[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    constructor_body = raw_text[start_idx + len(start_marker):end_idx]

    # Extract simple fields
    name = _extract_field(constructor_body, "name") or "Autonomous Pipeline"
    description = _extract_field(constructor_body, "description") or "Autonomous pipeline"
    output_dir = _extract_field(constructor_body, "output_dir") or "output"

    # Extract stages dict
    stages = _extract_stages(constructor_body)

    if not stages:
        raise ValueError("stages dict is empty or could not be extracted")

    return AutonomousPipelineSpec(
        name=name,
        description=description,
        stages=stages,
        output_dir=output_dir,
    )


def _extract_field(body: str, field_name: str) -> str | None:
    """Extract a simple string field from constructor body."""
    m = re.search(rf'{field_name}\s*=\s*["\']([^"\']*)["\']', body)
    if m:
        return m.group(1)
    return None


def _extract_stages(body: str) -> dict[str, AutonomousStage]:
    """Extract stages dict from constructor body."""
    import ast

    # Try to find the stages= dict literal
    m = re.search(r'stages\s*=\s*\{', body)
    if not m:
        return {}

    # Find the matching closing brace by counting braces
    start = m.end()
    depth = 1
    end = start
    for i in range(start, len(body)):
        if body[i] == '{':
            depth += 1
        elif body[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    stages_content = body[start:end]

    # Try multiple patterns to extract stage entries
    patterns = [
        r'"(\w+)"\s*:\s*AutonomousStage\([^)]+\)',
        r"'(\w+)'\\s*:\\s*AutonomousStage\\([^)]+\\)",
    ]

    stages: dict[str, AutonomousStage] = {}

    for pattern in patterns:
        stage_entries = re.findall(pattern, stages_content, re.DOTALL)
        if stage_entries:
            break
    else:
        return {}

    for stage_name in stage_entries:
        if isinstance(stage_name, tuple):
            stage_name = stage_name[0]

        # Extract the AutonomousStage for this stage
        pattern = rf'["\']{stage_name}["\']\s*:\s*AutonomousStage\(([^)]+)\)'
        m = re.search(pattern, stages_content, re.DOTALL)
        if not m:
            continue

        config_body = m.group(1)
        name = _extract_field(config_body, "name") or stage_name
        description = _extract_multiline_field(config_body, "description")
        if not description:
            description = _extract_field(config_body, "description") or f"Stage: {stage_name}"
        max_turns_str = _extract_field(config_body, "max_turns") or "10"
        try:
            max_turns = int(max_turns_str)
        except ValueError:
            max_turns = 10
        validation_command = _extract_field(config_body, "validation_command") or "true"

        stages[stage_name] = AutonomousStage(
            name=name,
            description=description,
            max_turns=max_turns,
            validation_command=validation_command,
        )

    return stages


def _extract_multiline_field(body: str, field_name: str) -> str | None:
    """Extract a multi-line string field (triple-quoted)."""
    for quote in ('"""', "'''"):
        pattern = rf'{field_name}\s*=\s*{quote}(.+?){quote}'
        m = re.search(pattern, body, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def ouroboros_fix(
    spec: AutonomousPipelineSpec,
    error: str,
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn=None,
) -> Result[AutonomousPipelineSpec, str]:
    """Apply ouroboros fix to the spec.
    
    G' = G (recursive invoke) with error context.
    The fix is just another model invocation, not a code patch.
    """
    # Add error context to feedback
    ctx = ctx.with_feedback(f"Previous error: {error}")
    ctx = ctx.with_domain(DomainSection(
        heading="Previous Generation (fix according to error)",
        content=str(spec),
    ))

    # Re-invoke with error context - G' = G
    raw_result = invoke_model(ctx, base_url, ask_fn)

    if isinstance(raw_result, Ok):
        return Ok(_parse_response(raw_result.value))
    return raw_result
