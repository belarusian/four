"""
Runtime functions for bash pipeline spoke generation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import cast

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

from ._types import BashPipelineSpec, BashStage

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


def invoke_model(
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return a BashPipelineSpec with bash source."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert at generating bash pipeline spokes for the four framework. "
            "Generate a bash script that implements a multi-stage pipeline using bash functions."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing BashPipelineSpec.\n"
            "See the docstring for the exact response format."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write a BashPipelineSpec(...) expression.",
            "Put the COMPLETE bash source code as a TRIPLE-QUOTED string in the source field.",
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


def _parse_response(raw_text: str) -> BashPipelineSpec:
    """Parse the model's Python constructor response into a spec."""
    start_marker = "BashPipelineSpec("
    start_idx = raw_text.find(start_marker)

    if start_idx < 0:
        raise ValueError("BashPipelineSpec constructor not found in response")

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
    name = _extract_field(constructor_body, "name") or "Bash Pipeline"
    description = _extract_field(constructor_body, "description") or "Bash pipeline"
    output_dir = _extract_field(constructor_body, "output_dir") or "output"
    bash_functions_dir = _extract_field(constructor_body, "bash_functions_dir") or "lib"
    source = _extract_multiline_field(constructor_body, "source")

    if not source:
        raise ValueError("source field is required and must be non-empty")

    # Extract stages dict
    stages_raw = _extract_stages(constructor_body)
    stages = cast(dict[str, BashStage], stages_raw)

    if not stages:
        raise ValueError("stages dict is empty or could not be extracted")

    return BashPipelineSpec(
        name=name,
        description=description,
        stages=stages,  # type: ignore
        output_dir=output_dir,
        bash_functions_dir=bash_functions_dir,
        source=source,
    )


def _extract_field(body: str, field_name: str) -> str | None:
    """Extract a simple string field from constructor body."""
    m = re.search(rf'{field_name}\s*=\s*["\']([^"\']*)["\']', body)
    if m:
        return m.group(1)
    return None


def _extract_stages(body: str) -> dict[str, BashStage]:
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

    # Try to parse as a dict literal
    try:
        stages_dict = ast.literal_eval("{" + stages_content + "}")
        if isinstance(stages_dict, dict):
            # Dict may contain BashStage instances or dicts
            result: dict[str, BashStage] = {}
            for name, stage_data in stages_dict.items():
                if isinstance(stage_data, dict):
                    result[name] = BashStage(
                        name=name,
                        description=stage_data.get("description", f"Stage: {name}"),
                        prompt=stage_data.get("prompt", f"Stage: {name}"),
                        model=stage_data.get("model", "granite4.1:8b"),
                    )
                elif isinstance(stage_data, BashStage):
                    result[name] = stage_data
                else:
                    # Try to parse as BashStage constructor string
                    logger.warning(f"Unknown stage data type: {type(stage_data)}")
            return result
    except (SyntaxError, ValueError):
        pass

    # Fallback: parse manually for BashStage(...) constructor
    stages: dict[str, BashStage] = {}
    # Match patterns like "stage_name": BashStage(...) or StageConfig(...)
    for pattern in [
        r'"(\w+)"\s*:\s*BashStage\([^)]+\)',
        r"'(\w+)'\\s*:\\s*BashStage\\([^)]+\\)",
        r'"(\w+)"\\s*:\\s*StageConfig\\([^)]*\\)',
    ]:
        stage_entries = re.findall(pattern, stages_content, re.DOTALL)
        if stage_entries:
            break
    else:
        return {}

    for stage_name in stage_entries:
        if isinstance(stage_name, tuple):
            stage_name = stage_name[0]

        # Extract the BashStage/StageConfig for this stage
        pattern = rf'["\']{stage_name}["\']\s*:\s*(?:BashStage|StageConfig)\(([^)]+)\)'
        m = re.search(pattern, stages_content, re.DOTALL)
        if not m:
            continue

        config_body = m.group(1)
        model = _extract_field(config_body, "model") or "granite4.1:8b"
        prompt = _extract_multiline_field(config_body, "prompt")
        if not prompt:
            prompt = _extract_field(config_body, "prompt") or f"Stage: {stage_name}"

        stages[stage_name] = BashStage(
            name=stage_name,
            description=f"Stage: {stage_name}",
            prompt=prompt,
            model=model,
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


def parse_spec(raw: str) -> Result[BashPipelineSpec, str]:
    """Parse raw model output into a BashPipelineSpec."""
    try:
        start_marker = "BashPipelineSpec("
        start_idx = raw.find(start_marker)

        if start_idx < 0:
            return Err("BashPipelineSpec constructor not found in response")

        # Extract the constructor body by balancing parens
        depth = 0
        end_idx = start_idx
        for i in range(start_idx, len(raw)):
            ch = raw[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        constructor_body = raw[start_idx + len(start_marker):end_idx]

        # Extract simple fields
        name = _extract_field(constructor_body, "name") or "Bash Pipeline"
        description = _extract_field(constructor_body, "description") or "Bash pipeline"
        output_dir = _extract_field(constructor_body, "output_dir") or "output"
        bash_functions_dir = _extract_field(constructor_body, "bash_functions_dir") or "lib"
        source = _extract_multiline_field(constructor_body, "source")

        if not source:
            return Err("source field is required and must be non-empty")

        # Extract stages dict
        stages_raw = _extract_stages(constructor_body)
        stages = cast(dict[str, BashStage], stages_raw)

        if not stages:
            return Err("stages dict is empty or could not be extracted")

        return Ok(BashPipelineSpec(
            name=name,
            description=description,
            stages=stages,
            output_dir=output_dir,
            bash_functions_dir=bash_functions_dir,
            source=source,
        ))
    except (SyntaxError, ValueError, AttributeError) as e:
        return Err(f"Parse error: {e}")


def ouroboros_fix(
    spec: BashPipelineSpec,
    error: str,
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn=None,
) -> Result[BashPipelineSpec, str]:
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
        return parse_spec(raw_result.value)
    return raw_result
