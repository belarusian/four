"""IO boundaries for the orchestrator generator.

G_orchestrator: invoke model, return OrchestratorSpec
V_orchestrator: validate spec instance
G'_orchestrator: patch-based ouroboros
IO: emit orchestrator directory with idea-forge templates + spokes
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
from ._types import OrchestratorSpec, ManifestEntry

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


def invoke_model(
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return an OrchestratorSpec."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert orchestrator developer. "
            "Generate a complete orchestrator system for distributed LLM agents."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing OrchestratorSpec.\n"
            "See the docstring for the exact response format."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write an OrchestratorSpec(...) expression.",
            "Use triple quotes: description=\"\"\"...\"\"\" and summary=\"\"\"...\"\"\"",
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


def _parse_response(raw_text: str) -> OrchestratorSpec:
    """Parse the model's Python constructor response into a spec."""
    start_marker = "OrchestratorSpec("
    start_idx = raw_text.find(start_marker)

    if start_idx < 0:
        raise ValueError("OrchestratorSpec constructor not found in response")

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
    name = _extract_field(constructor_body, "name") or "orchestrator"
    description = _extract_multiline_field(constructor_body, "description") or "Orchestrator system"

    # Extract booleans
    email_integration = "email_integration=True" in constructor_body
    github_workflow = "github_workflow=True" in constructor_body

    # Extract spoke types
    spoke_types = _extract_spoke_types(constructor_body)

    # Extract manifest entries
    manifest_entries = _extract_manifest_entries(constructor_body)

    return OrchestratorSpec(
        name=name,
        description=description,
        email_integration=email_integration,
        github_workflow=github_workflow,
        spoke_types=spoke_types,
        manifest_entries=tuple(manifest_entries),
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


def _extract_spoke_types(body: str) -> tuple[str, ...]:
    """Extract spoke types tuple."""
    m = re.search(r'spoke_types\s*=\s*\(([^)]*)\)', body, re.DOTALL)
    if not m:
        return ("bash", "bash_toolcall", "issue_agent")
    
    content = m.group(1)
    types = re.findall(r'"(\w+)"', content)
    return tuple(types) if types else ("bash", "bash_toolcall", "issue_agent")


def _extract_manifest_entries(body: str) -> list[ManifestEntry]:
    """Extract manifest entries from constructor body."""
    # Look for manifest_entries= tuple
    m = re.search(r'manifest_entries\s*=\s*\(([^)]*)\)', body, re.DOTALL)
    if not m:
        return []
    
    content = m.group(1)
    entries = []
    
    # Extract ManifestEntry(...) patterns
    entry_pattern = r'ManifestEntry\(([^)]+)\)'
    for entry_match in re.finditer(entry_pattern, content, re.DOTALL):
        entry_body = entry_match.group(1)
        repo = _extract_field(entry_body, "repo") or "unnamed"
        title = _extract_field(entry_body, "title") or "Unnamed"
        description = _extract_multiline_field(entry_body, "description") or ""
        templates = _extract_templates(entry_body)
        entries.append(ManifestEntry(
            repo=repo,
            title=title,
            description=description,
            templates=templates,
        ))
    
    return entries


def _extract_templates(body: str) -> tuple[str, ...]:
    """Extract templates tuple from manifest entry."""
    m = re.search(r'templates\s*=\s*\(([^)]*)\)', body, re.DOTALL)
    if not m:
        return ("python-research",)
    
    content = m.group(1)
    templates = re.findall(r'"(\w+)"', content)
    return tuple(templates) if templates else ("python-research",)
