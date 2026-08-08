"""
Validation for autonomous pipeline generation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from four.generators._types import Cycle, Err, Ok, Result

from ._types import AutonomousPipelineSpec, AutonomousStage


def validate_spec_instance(spec: AutonomousPipelineSpec) -> Result[AutonomousPipelineSpec, str]:
    """Validate the spec instance has required fields."""
    if not spec.name or not spec.name.strip():
        return Err("Pipeline name is required")
    if not spec.description or not spec.description.strip():
        return Err("Pipeline description is required")
    if not spec.stages or len(spec.stages) < 1:
        return Err("Pipeline must have at least 1 stage")
    for stage_name, stage in spec.stages.items():
        if not stage_name or not stage_name.strip():
            return Err(f"Stage {stage_name}: name is required")
        if not stage.description or not stage.description.strip():
            return Err(f"Stage {stage_name}: description is required")
        if stage.max_turns < 1:
            return Err(f"Stage {stage_name}: max_turns must be >= 1")
        if stage.max_turns > 100:
            return Err(f"Stage {stage_name}: max_turns too high (max 100)")
    return Ok(spec)


def validate_stage_names_unique(stages: dict) -> Result[None, str]:
    """Ensure all stage names are unique and valid."""
    names = list(stages.keys())
    if len(names) != len(set(names)):
        return Err("Duplicate stage names detected")
    for name in names:
        if not re.match(r"^[a-z][a-z0-9_-]*$", name):
            return Err(f"Invalid stage name: {name} (must be lowercase, alphanumeric with -/_)")
    return Ok(None)


def validate_pipeline_structure(spec: AutonomousPipelineSpec) -> Result[AutonomousPipelineSpec, str]:
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
