"""Orchestrator generator context builder.

Injects the orchestrator architecture and pattern into the model's system prompt.
"""

from __future__ import annotations

from four.generators._types import DomainSection, GenerationContext

_ORCHESTRATOR_ARCHITECTURE = """\
## Orchestrator Architecture

The orchestrator is a system that manages work flow from intake to delivery:

```
Email/Intake → GitHub Issues → Auto-Dispatch → Spokes → GitHub Lifecycle
```

### Key Components

1. **Intake Layer** (email-integration.ts)
   - IMAP/PGP for email intake
   - Issue creation from emails
   - PGP signature verification

2. **Orchestrator** (orchestrator/*.ts)
   - PostgreSQL state machine for tasks
   - Profile-scoring collaborators
   - GitHub lifecycle management
   - No local agent spawning (all through spokes)

3. **Spokes** (coworker/*.py)
   - Python-based agent execution
   - Register with orchestrator
   - Poll for tasks
   - Execute via four's G→V→G' loop
   - Report results

4. **Manifest** (manifests/*.tsv)
   - Repository definitions
   - Template assignments
   - Profile assignments

### Design Rules

1. **No local agent spawning** - All agents run through spokes
2. **PRs only** - No manual remote code changes
3. **LLM JSON extraction** - No regex for parsing
4. **Tests required** - All LLM code must have tests

## Spoke Types

- **bash** - Simple bash commands with regex parsing
- **bash_toolcall** - Bash via tool calls with structured parsing
- **issue_agent** - GitHub issue triage and assignment
- **four_super_spoke** - Full four-agent with tool calls + auto-retry
"""

_MANIFEST_EXAMPLE = """\
## Manifest Format

manifests/projects.tsv:
```
repo\ttitle\tdescription\ttemplates\tstarter_pack\tsummary
agent-system\tAgent System\tLLM agent orchestration\tpython-research,notebook-laboratory\t\tAgent System is a distributed LLM agent platform
```

Columns:
- repo: Repository name
- title: Short description
- description: Long description
- templates: Comma-separated template names
- starter_pack: Optional starter pack name
- summary: One-line summary
"""

_GENERATION_PATTERN = """\
## Generation Pattern

The generator produces:

1. **Project structure** via idea-forge templates:
   - pyproject.toml
   - Makefile
   - src/, tests/, docs/ directories
   - .github/ workflows, templates

2. **Spoke implementations**:
   - coworker/ directory with spoke Python files
   - Each spoke implements four's G→V→G' loop
   - Register, poll, heartbeat, result endpoints

3. **Orchestrator code**:
   - src/orchestrator/*.ts
   - Email intake, GitHub integration
   - Task management, profile scoring

4. **Manifest entries** for all repos

## Template Selection

For orchestrators, combine templates:
- **python-research** - Python project structure
- **notebook-laboratory** - For experimentation
- **static-website** - For documentation
- **rust-library** - For CLI tools (if needed)
"""

def build_orchestrator_context() -> GenerationContext:
    """Build context for the orchestrator generator."""
    return GenerationContext(
        domain_context=(
            DomainSection("Orchestrator Architecture", _ORCHESTRATOR_ARCHITECTURE),
            DomainSection("Manifest Format", _MANIFEST_EXAMPLE),
            DomainSection("Generation Pattern", _GENERATION_PATTERN),
        ),
        available_packages="pathlib, os, typing",
        default_task=(
            "Generate an orchestrator system for email-driven GitHub workflow. "
            "Include project structure, spokes, and manifest entries."
        ),
    )
