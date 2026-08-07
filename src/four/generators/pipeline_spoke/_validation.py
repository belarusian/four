"""Validation for pipeline spoke specs."""

from __future__ import annotations

from four.generators._types import Err, Ok, Result

from ._types import PipelineSpokeSpec


def validate_spec_instance(spec: PipelineSpokeSpec) -> Result[PipelineSpokeSpec, str]:
    """Validate a PipelineSpokeSpec instance."""
    errors: list[str] = []

    if not spec.name:
        errors.append("name must be non-empty")

    if not spec.description:
        errors.append("description must be non-empty")

    if not spec.stages:
        errors.append("stages must be non-empty")

    required_stages = ["outline", "draft", "review", "revision"]
    for stage in required_stages:
        if stage not in spec.stages:
            errors.append(f"stages must include '{stage}'")

    for stage_name, stage_config in spec.stages.items():
        if not stage_config.model:
            errors.append(f"stages['{stage_name}'].model must be non-empty")
        if not stage_config.prompt:
            errors.append(f"stages['{stage_name}'].prompt must be non-empty")

    if errors:
        return Err("; ".join(errors))
    return Ok(spec)
