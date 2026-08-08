"""
Prompt building for autonomous pipeline generation.

Sources from three repos as memory and state:
- four: the algebra (G, V1, V2, emit)
- experiments: bash pipeline patterns (essay-pipeline.sh, icepick.sh)
- idea-forge: project structure (templates, manifests)
"""

from __future__ import annotations

from pathlib import Path

from four.generators._types import (
    DomainSection,
    GenerationContext,
)


def _read_repo(repo_path: Path) -> str:
    """Read key files from a repo as context."""
    if not repo_path.exists():
        return f"Repo not found: {repo_path}"

    parts = [f"## Repo: {repo_path.name}\n"]

    # Read README if it exists
    readme = repo_path / "README.md"
    if readme.exists():
        parts.append(f"### README.md\n```\n{readme.read_text()[:2000]}\n```\n")

    # Read key scripts
    for pattern in ["*.sh", "*.py"]:
        for f in sorted(repo_path.glob(pattern)):
            if f.name.endswith("__pycache__"):
                continue
            content = f.read_text()
            if len(content) > 1500:
                content = content[:1500] + "\n... (truncated)"
            parts.append(f"### {f.name}\n```\n{content}\n```\n")

    return "\n".join(parts)


def build_autonomous_pipeline_context() -> GenerationContext:
    """Build the generation context by sourcing from our repos."""
    # Find the repos relative to this file
    four_root = Path(__file__).parent.parent.parent.parent.parent  # /Users/av4nda/Research/four

    domain_sections = [
        DomainSection(
            heading="Source: four algebra",
            content=_read_repo(four_root / "src" / "four"),
        ),
        DomainSection(
            heading="Source: experiments (bash pipelines)",
            content=_read_repo(Path("/Users/av4nda/AI/experiments")),
        ),
        DomainSection(
            heading="Source: idea-forge (structure)",
            content=_read_repo(Path("/Users/av4nda/AI/idea-forge")),
        ),
        DomainSection(
            heading="Autonomous Pipeline Pattern",
            content="""
The autonomous pipeline is a single four.core.run() call with:
- G: litellm_invoke (LLM)
- V1: regex_parse (extract bash commands)
- V2: local_env (execute bash - git, tests, writes)
- emit: save_trajectory (persist)
- system: tells the agent what to do (source from experiments for patterns)
- prompt: the goal (ticket, feature, refactor)
- max_steps: how many turns before stopping

The agent uses bash to:
- Write/modify files
- Run git add/commit/push
- Execute tests/linters
- Call other LLMs via curl if needed

Source the patterns from the experiments repo. Source the structure from idea-forge. Source the algebra from four.
""",
        ),
    ]

    ctx = GenerationContext(domain_context=tuple(domain_sections))
    return ctx
