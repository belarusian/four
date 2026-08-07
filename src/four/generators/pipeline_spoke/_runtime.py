"""IO boundaries for the pipeline spoke generator.

G_pipeline: invoke model, return PipelineSpokeSpec
V_pipeline: validate spec instance
G'_pipeline: patch-based ouroboros
IO: emit .py spoke file
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from four.generators._types import (
    AskFn,
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

from . import _types
from ._types import PipelineSpokeSpec, StageConfig

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


def invoke_model(
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return a PipelineSpokeSpec."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert spoke developer. Generate a pipeline spoke that "
            "implements a multi-stage workflow like the experiments essay pipeline. "
            "The spoke registers with the orchestrator, polls for tasks, executes "
            "pipeline stages in order, and reports results."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing PipelineSpokeSpec.\n"
            "See the docstring for the exact response format."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write a PipelineSpokeSpec(...) expression.",
            "Put the COMPLETE spoke source code as a TRIPLE-QUOTED string in the source field.",
            "Use triple quotes: source=\"\"\"...\"\"\"",
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


def _parse_response(raw_text: str) -> PipelineSpokeSpec:
    """Parse the model's Python constructor response into a spec."""
    start_marker = "PipelineSpokeSpec("
    start_idx = raw_text.find(start_marker)

    if start_idx < 0:
        raise ValueError("PipelineSpokeSpec constructor not found in response")

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
    name = _extract_field(constructor_body, "name") or "Pipeline Spoke"
    description = _extract_field(constructor_body, "description") or "Pipeline spoke"
    output_dir = _extract_field(constructor_body, "output_dir") or "output"

    # Extract stages dict
    stages = _extract_stages(constructor_body)

    if not stages:
        raise ValueError("stages dict is empty or could not be extracted")

    return PipelineSpokeSpec(
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


def _extract_stages(body: str) -> dict[str, StageConfig]:
    """Extract stages dict from constructor body."""
    import ast

    # Try to find the stages= dict literal
    m = re.search(r'stages\s*=\s*\{([^}]*)\}', body, re.DOTALL)
    if not m:
        return {}

    stages_content = m.group(1)

    # Parse stage entries: "stage_name": StageConfig(...)
    stage_entries = re.findall(
        r'"(\w+)"\s*:\s*StageConfig\([^)]+\)',
        stages_content,
        re.DOTALL,
    )

    stages: dict[str, StageConfig] = {}

    for stage_name in stage_entries:
        # Extract the StageConfig for this stage
        pattern = rf'"{stage_name}"\s*:\s*StageConfig\(([^)]+)\)'
        m = re.search(pattern, stages_content)
        if not m:
            continue

        config_body = m.group(1)
        model = _extract_field(config_body, "model") or "granite4.1:3b"

        # Extract prompt - handle multi-line strings
        prompt = _extract_multiline_field(config_body, "prompt")
        if not prompt:
            prompt = f"Pipeline stage: {stage_name}"

        stages[stage_name] = StageConfig(model=model, prompt=prompt)

    return stages


def _extract_multiline_field(body: str, field_name: str) -> str | None:
    """Extract a multi-line string field (triple-quoted)."""
    for quote in ('"""', "'''"):
        pattern = rf'{field_name}\s*=\s*{quote}(.+?){quote}'
        m = re.search(pattern, body, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None
