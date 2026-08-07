"""Validation for idea-forge template specs."""

from __future__ import annotations

from four.generators._types import Err, Ok, Result

from ._types import IdeaForgeTemplate


def validate_spec_instance(spec: IdeaForgeTemplate) -> Result[IdeaForgeTemplate, str]:
    """Validate an IdeaForgeTemplate instance."""
    errors: list[str] = []

    if not spec.name:
        errors.append("name must be non-empty")

    if not spec.description:
        errors.append("description must be non-empty")

    if not spec.readme:
        errors.append("readme must be non-empty")

    if not spec.template_env:
        errors.append("template_env must be non-empty")

    if errors:
        return Err("; ".join(errors))
    return Ok(spec)
