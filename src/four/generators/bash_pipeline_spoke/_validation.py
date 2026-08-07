"""
Validation for bash pipeline spoke generation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from four.generators._types import Cycle, Err, Ok, Result

from ._types import BashPipelineSpec


def validate_spec_instance(spec: BashPipelineSpec) -> Result[BashPipelineSpec, str]:
    """Validate the spec instance has required fields."""
    if not spec.name or not spec.name.strip():
        return Err("Pipeline name is required")
    if not spec.description or not spec.description.strip():
        return Err("Pipeline description is required")
    if not spec.source or not spec.source.strip():
        return Err("Pipeline source is required")
    if not spec.stages or len(spec.stages) < 2:
        return Err("Pipeline must have at least 2 stages")
    for stage_name, stage in spec.stages.items():
        if not stage_name or not stage_name.strip():
            return Err(f"Stage {stage_name}: name is required")
        if not stage.prompt or not stage.prompt.strip():
            return Err(f"Stage {stage_name}: prompt is required")
        if not stage.model or not stage.model.strip():
            return Err(f"Stage {stage_name}: model is required")
    return Ok(spec)


def validate_bash_syntax(source: str) -> Result[None, str]:
    """Validate that the bash source has valid syntax."""
    try:
        result = subprocess.run(
            ["bash", "-n", "-c", source],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Ok(None)
        return Err(f"Syntax error: {result.stderr.strip()}")
    except FileNotFoundError:
        return Err("bash not found - skipping syntax validation")
    except subprocess.TimeoutExpired:
        return Err("bash syntax validation timed out")


def validate_stage_names_unique(stages: dict) -> Result[None, str]:
    """Ensure all stage names are unique and valid."""
    names = list(stages.keys())
    if len(names) != len(set(names)):
        return Err("Duplicate stage names detected")
    for name in names:
        if not re.match(r"^[a-z][a-z0-9_-]*$", name):
            return Err(f"Invalid stage name: {name} (must be lowercase, alphanumeric with -/_")
    return Ok(None)


def validate_pipeline_structure(spec: BashPipelineSpec) -> Result[BashPipelineSpec, str]:
    """Validate the complete pipeline structure."""
    # Check spec instance
    match validate_spec_instance(spec):
        case Ok(_) as ok:
            pass
        case Err(e) as err:
            return err

    # Check stage names
    match validate_stage_names_unique(spec.stages):
        case Ok(_) as ok:
            pass
        case Err(e) as err:
            return err

    return Ok(spec)
