"""Idea-forge template generator context builder.

Injects idea-forge template structure and examples into the model's system prompt.
"""

from __future__ import annotations

from four.generators._types import DomainSection, GenerationContext

_IDEA_FORGE_DOCS = """\
## Idea Forge Template Structure

Templates are stored in idea-forge/templates/{name}/

Each template requires:
- README.md: Description of the template
- template.env: TEMPLATE_ID={name}

Templates are composable - multiple templates can be combined.

## Example Template: python-research

```
templates/python-research/
├── README.md
└── template.env
```

README.md:
```markdown
# python-research

Composable repository template used by idea-forge.
```

template.env:
```bash
TEMPLATE_ID=python-research
```

## Available Template Types

- python-research: Python project with testing, docs, CI
- rust-library: Rust library with Cargo.toml
- rust-workspace: Rust workspace with multiple crates
- c-library: C library with Makefile
- c-kernel: C kernel project structure
- latex-book: LaTeX book template
- latex-paper: LaTeX paper template
- javascript-webapp: JavaScript web application
- notebook-laboratory: Jupyter notebook environment
- static-website: Static site with Hugo/Markdown

## Template Composability

Templates can be combined. Example manifest entry:
```
math-machines\tMath Machines\tComputational math research\tpython-research,notebook-laboratory
```

This creates a repository with both Python research structure AND notebook support.
"""

_EXAMPLE_TEMPLATES = """\
## Example Templates

### python-research
```bash
TEMPLATE_ID=python-research
```
Creates: src/, tests/, benchmarks/, docs/, papers/, pyproject.toml, Makefile

### rust-library
```bash
TEMPLATE_ID=rust-library
```
Creates: src/, Cargo.toml, tests/, examples/, README

### notebook-laboratory
```bash
TEMPLATE_ID=notebook-laboratory
```
Creates: notebooks/, data/, requirements.txt, environment.yml
"""

def build_idea_forge_context() -> GenerationContext:
    """Build context for the idea-forge template generator."""
    return GenerationContext(
        domain_context=(
            DomainSection("Idea Forge Template Structure", _IDEA_FORGE_DOCS),
            DomainSection("Example Templates", _EXAMPLE_TEMPLATES),
        ),
        available_packages="pathlib, os",
        default_task=(
            "Generate an idea-forge template that creates a composable repository structure. "
            "Return README.md and template.env content."
        ),
    )
