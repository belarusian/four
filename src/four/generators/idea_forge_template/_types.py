"""Idea-forge template generator types.

A template defines a composable repository structure for idea-forge.

Template structure:
- README.md: Description of the template
- template.env: Template ID

Composable templates can be combined. Example: python-research + notebook-laboratory.

Response format:

IdeaForgeTemplate(
    name="python-research",
    description="Python research project template with testing and documentation",
    readme="# python-research\n\nComposable repository template for Python research projects.",
    template_env="TEMPLATE_ID=python-research",
)

For composable templates, multiple templates can be generated together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class IdeaForgeTemplate:
    """Complete idea-forge template specification."""
    name: str
    description: str
    readme: str
    template_env: str
