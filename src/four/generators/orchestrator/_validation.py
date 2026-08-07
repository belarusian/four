"""Validation for orchestrator specs."""

from __future__ import annotations

from four.generators._types import Err, Ok, Result

from ._types import OrchestratorSpec, ManifestEntry


def validate_spec_instance(spec: OrchestratorSpec) -> Result[OrchestratorSpec, str]:
    """Validate an OrchestratorSpec instance."""
    errors: list[str] = []

    if not spec.name:
        errors.append("name must be non-empty")

    if not spec.description:
        errors.append("description must be non-empty")

    if not spec.spoke_types:
        errors.append("spoke_types must be non-empty")

    for entry in spec.manifest_entries:
        if not entry.repo:
            errors.append("manifest entries must have repo name")
        if not entry.title:
            errors.append("manifest entries must have title")

    if errors:
        return Err("; ".join(errors))
    return Ok(spec)
