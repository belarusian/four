"""
Types for autonomous pipeline generation.

An autonomous pipeline is a long-running system that:
- Reads a high-level goal (e.g., "Implement ticket X with 1000 commits")
- Generates PRs and commits autonomously
- Validates each step via bash (git, tests, linters)
- Uses four.core.run() with bash tool for code generation
- Loops until the goal is achieved or max_steps is reached
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class AutonomousStage:
    """A single autonomous stage that can run multiple turns."""
    name: str
    description: str
    max_turns: int = 10
    validation_command: str = "true"


@dataclass(frozen=True)
class AutonomousPipelineSpec:
    """Complete autonomous pipeline specification.
    
    This generates a Python script that runs an autonomous pipeline:
    - Takes a goal/ticket description as input
    - Generates code and commits via four.core.run() with bash tool
    - Validates each step with bash commands
    - Loops until goal achieved or max_steps reached
    
    Response format -- Python constructor:
    
    AutonomousPipelineSpec(
        name="autonomous-dev-pipeline",
        description="Autonomous development pipeline for ticket implementation",
        stages={
            "analyze": AutonomousStage(
                name="analyze",
                description="Analyze ticket requirements",
                max_turns=5,
                validation_command="test -f output/analysis.md",
            ),
            "plan": AutonomousStage(
                name="plan",
                description="Create implementation plan",
                max_turns=3,
                validation_command="test -f output/plan.md",
            ),
            "implement": AutonomousStage(
                name="implement",
                description="Write code and commit",
                max_turns=20,
                validation_command="make test",
            ),
        },
        output_dir="output/{topic-slug}",
    )
    """
    name: str
    description: str
    stages: Dict[str, AutonomousStage]
    output_dir: str = "output"
