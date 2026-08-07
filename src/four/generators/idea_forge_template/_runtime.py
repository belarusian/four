"""IO boundaries for the idea-forge template generator.

G_template: invoke model, return IdeaForgeTemplate
V_template: validate spec instance
G'_template: patch-based ouroboros
IO: emit template directory with README.md and template.env
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from four.generators._types import (
    AskFn,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from four.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)

from . import _types
from ._types import IdeaForgeTemplate

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


def invoke_model(
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return an IdeaForgeTemplate."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert repository template developer for idea-forge. "
            "Generate composable templates that create consistent, well-documented "
            "research and software projects."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing IdeaForgeTemplate.\n"
            "See the docstring for the exact response format."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write an IdeaForgeTemplate(...) expression.",
            "Use triple quotes: readme=\"\"\"...\"\"\" and template_env=\"\"\"...\"\"\"",
            "No markdown fencing.",
        ),
    )

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    try:
        spec = _parse_response(raw_text)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


def _parse_response(raw_text: str) -> IdeaForgeTemplate:
    """Parse the model's Python constructor response into a spec."""
    start_marker = "IdeaForgeTemplate("
    start_idx = raw_text.find(start_marker)

    if start_idx < 0:
        raise ValueError("IdeaForgeTemplate constructor not found in response")

    # Extract the constructor body by balancing parens
    depth = 0
    end_idx = start_idx
    for i in range(start_idx, len(raw_text)):
        ch = raw_text[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    constructor_body = raw_text[start_idx + len(start_marker):end_idx]

    # Extract simple fields
    name = _extract_field(constructor_body, "name") or "idea-forge-template"
    description = _extract_field(constructor_body, "description") or "Idea Forge template"

    # Extract readme and template_env as triple-quoted strings
    readme = _extract_multiline_field(constructor_body, "readme")
    if not readme:
        readme = f"# {name}\n\nComposable repository template for {description.lower()}."

    template_env = _extract_multiline_field(constructor_body, "template_env")
    if not template_env:
        template_env = f"TEMPLATE_ID={name}"

    return IdeaForgeTemplate(
        name=name,
        description=description,
        readme=readme,
        template_env=template_env,
    )


def _extract_field(body: str, field_name: str) -> str | None:
    """Extract a simple string field from constructor body."""
    m = re.search(rf'{field_name}\s*=\s*["\']([^"\']*)["\']', body)
    if m:
        return m.group(1)
    return None


def _extract_multiline_field(body: str, field_name: str) -> str | None:
    """Extract a multi-line string field (triple-quoted)."""
    for quote in ('"""', "'''"):
        pattern = rf'{field_name}\s*=\s*{quote}(.+?){quote}'
        m = re.search(pattern, body, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None
