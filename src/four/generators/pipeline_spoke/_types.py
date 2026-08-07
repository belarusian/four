"""Pipeline spoke generator types.

A pipeline spoke implements a multi-stage workflow:
- Topic → Outline → Draft → Review → Final
- Each stage uses a specific model
- Each stage saves its output to disk
- The pipeline is persistent via filesystem

Response format:

PipelineSpokeSpec(
    name="essay-pipeline",
    description="Multi-stage essay writing pipeline",
    stages={
        "outline": StageConfig(
            model="granite4.1:3b",
            prompt="You are designing a rigorous conceptual essay...",
        ),
        "draft": StageConfig(
            model="granite4.1:3b",
            prompt="Write a substantial analytical essay from the supplied topic and outline...",
        ),
        "review": StageConfig(
            model="granite4.1:8b",
            prompt="Act as a demanding reviewer of the essay below...",
        ),
        "revision": StageConfig(
            model="granite4.1:3b",
            prompt="Revise the supplied essay in response to the review...",
        ),
    },
    output_dir="output/{topic-slug}",
)

The spoke uses these stages in order, passing the output of each to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single pipeline stage."""
    model: str
    prompt: str


@dataclass(frozen=True)
class PipelineSpokeSpec:
    """Complete pipeline spoke specification.

    A spoke that executes a multi-stage pipeline where each stage
    uses a different model and passes its output to the next stage.

    Response format -- Python constructor:

    PipelineSpokeSpec(
        name="essay-pipeline",
        description="Multi-stage essay writing pipeline",
        stages={
            "outline": StageConfig(
                model="granite4.1:3b",
                prompt="You are designing a rigorous conceptual essay...",
            ),
            "draft": StageConfig(...),
            "review": StageConfig(...),
            "revision": StageConfig(...),
        },
        output_dir="output/{topic-slug}",
    )
    """
    name: str
    description: str
    stages: dict[str, StageConfig]
    output_dir: str
