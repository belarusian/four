"""Orchestrator generator types.

An orchestrator is a system that:
1. Receives work via email/intake layer
2. Creates GitHub issues from work items
3. Auto-dispatches tasks to LLM spokes
4. Manages GitHub lifecycle (issues → PRs → reviews → merge)
5. No local agent spawning (all through spokes)

The orchestrator structure:
- src/orchestrator/ - TypeScript orchestration logic
- coworker/ - Spoke implementations
- manifests/ - Task and project manifests
- docs/ - Documentation

Response format:

OrchestratorSpec(
    name="compsci-boutique",
    description="Email-driven GitHub workflow orchestrator",
    email_integration=True,
    github_workflow=True,
    spoke_types=["bash", "bash_toolcall", "issue_agent"],
    manifest_entries=[
        ManifestEntry(repo="agent-system", templates=["python-research"]),
    ],
)

The generated orchestrator uses idea-forge templates for structure
and generates spoke code for each agent type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ManifestEntry:
    """A manifest entry for a repository in the orchestrator."""
    repo: str
    title: str
    description: str
    templates: tuple[str, ...]


@dataclass(frozen=True)
class OrchestratorSpec:
    """Complete orchestrator specification."""
    name: str
    description: str
    email_integration: bool
    github_workflow: bool
    spoke_types: tuple[str, ...]
    manifest_entries: tuple[ManifestEntry, ...]
