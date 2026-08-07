"""
Prompt building for bash pipeline spoke generation.
"""

from __future__ import annotations

from pathlib import Path

from four.generators._types import (
    DomainSection,
    GenerationContext,
)


def build_bash_pipeline_context() -> GenerationContext:
    """Build the generation context for bash pipeline spokes."""
    types_source = (Path(__file__).parent / "_types.py").read_text()

    domain_sections = [
        DomainSection(
            heading="Bash Pipeline Spoke Requirements",
            content="""
A bash pipeline spoke implements a multi-stage workflow using bash functions.
Each stage:
- Takes input from previous stage (or user prompt for first stage)
- Calls an LLM with a specific prompt
- Saves output to a markdown file
- Passes output to next stage

The pipeline should follow the four-function algebra pattern:
- Each stage is independent and testable
- Stages chain together via file-based output
- Error handling is explicit at each stage
""",
        ),
        DomainSection(
            heading="Existing Artifacts",
            content="No existing artifacts to modify.",
        ),
    ]

    ctx = GenerationContext(domain_context=tuple(domain_sections))

    return ctx
