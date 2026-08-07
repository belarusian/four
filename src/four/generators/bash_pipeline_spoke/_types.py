"""
Types for bash pipeline spoke generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class BashStage:
    """A single stage in a bash pipeline."""
    name: str
    description: str
    prompt: str
    model: str = "granite4.1:8b"


@dataclass(frozen=True)
class BashPipelineSpec:
    """Specification for a bash pipeline spoke."""
    name: str
    description: str
    stages: Dict[str, BashStage]
    output_dir: str = "output"
    bash_functions_dir: str = "lib"
    source: str = ""
