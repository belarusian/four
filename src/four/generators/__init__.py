"""Generators framework — tools that create tools.

This package provides the generation loop and core types used by
domain-specific generators (e.g., four_agent, letsplot_notebook).
"""

from four.generators._types import Result, Ok, Err, GenerationContext, DomainSection
from four.generators._loop import generation_loop

__all__ = [
    "Result",
    "Ok",
    "Err",
    "GenerationContext",
    "DomainSection",
    "generation_loop",
]
