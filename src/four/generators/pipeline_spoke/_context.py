"""Pipeline spoke generator context builder.

Injects the experiments pipeline structure and stage configurations
into the model's system prompt.
"""

from __future__ import annotations

from four.generators._types import DomainSection, GenerationContext

_EXPERIMENTS_PIPELINE = """\
## Experiments Pipeline (Essay as a Pipeline)

The experiments pipeline implements a 4-stage workflow:

```
topic
  ↓ (outline stage, granite4.1:3b)
outline
  ↓ (draft stage, granite4.1:3b)
draft
  ↓ (review stage, granite4.1:8b)
review
  ↓ (revision stage, granite4.1:3b)
final
```

Each stage:
- Takes input from the previous stage (or topic for first stage)
- Uses a specific model (3B for generation, 8B for review)
- Saves output to disk
- Passes output to next stage

## Pipeline Stages

### Outline Stage
- Input: topic
- Model: granite4.1:3b
- Prompt: Create a coherent essay outline with central argument, thesis, sections, objections, consequences
- Output: outline.md

### Draft Stage  
- Input: topic + outline
- Model: granite4.1:3b
- Prompt: Write analytical essay from outline, develop paragraphs, define concepts
- Output: draft.md

### Review Stage
- Input: draft
- Model: granite4.1:8b (stronger for critical judgment)
- Prompt: Identify conceptual gaps, unsupported transitions, ambiguities, weak definitions
- Output: review.md

### Revision Stage
- Input: draft + review
- Model: granite4.1:3b
- Prompt: Revise essay in response to review, preserve good content, fix problems
- Output: essay.md

## Spoke Behavior

The generated spoke should:
1. Register with orchestrator at startup
2. Poll for tasks (each task has a "topic" field)
3. For each topic, execute the 4-stage pipeline
4. Save each stage's output to disk (output/{topic-slug}/{stage}.md)
5. Report completion with final essay path
6. Continue polling for more tasks

## File Output

Each run creates a directory: output/{slugified-topic}/

Files:
- topic.txt (input topic)
- outline.md (stage 1 output)
- draft.md (stage 2 output)  
- review.md (stage 3 output)
- essay.md (final output)

"""

_EXPERIMENTS_CONFIG = """\
## Configuration

Environment variables (with defaults):
- DRAFT_MODEL: "granite4.1:3b" (stage 1, 2, 4)
- REVIEW_MODEL: "granite4.1:8b" (stage 3)
- OUTPUT_DIR: "output"
- LOG_DIR: "logs"

Models are run via: ollama run {model}

Pipeline is defined by prompts in prompts/ directory:
- prompts/outline.txt
- prompts/draft.txt
- prompts/review.txt
- prompts/revise.txt

Each prompt uses {{KEY}} variable substitution:
- outline: {{TOPIC}}
- draft: {{TOPIC}} {{OUTLINE}}
- review: {{ESSAY}}
- revise: {{ESSAY}} {{REVIEW}}

Prompt rendering is done by scripts/render-prompt.py
"""

def build_pipeline_context() -> GenerationContext:
    """Build context for the pipeline spoke generator."""
    return GenerationContext(
        domain_context=(
            DomainSection("Experiments Pipeline", _EXPERIMENTS_PIPELINE),
            DomainSection("Configuration", _EXPERIMENTS_CONFIG),
        ),
        available_packages="requests, urllib3, pathlib",
        default_task=(
            "Generate a pipeline spoke that implements the experiments essay pipeline: "
            "topic → outline → draft → review → revision with appropriate models."
        ),
    )
