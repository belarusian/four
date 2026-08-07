"""Pipeline spoke generator.

Usage:
    python -m four.generators.pipeline_spoke.generate \\
        --prompt "Create an essay pipeline spoke" \\
        --output-dir ./spokes

    python -m four.generators.pipeline_spoke.generate --live
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from four.generators._types import (
    DomainSection,
    Err,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from four.generators._loop import generation_loop, result_to_exit

from ._types import PipelineSpokeSpec, StageConfig
from ._context import build_pipeline_context
from ._runtime import _TYPES_SOURCE, invoke_model
from ._validation import validate_spec_instance

logger = logging.getLogger(__name__)


def _make_invoke(base_url: str = "", ask_fn=None):
    """G_pipeline: Context -> Result[PipelineSpokeSpec]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, base_url, ask_fn)
    return invoke


def _make_parse():
    """V1: PipelineSpokeSpec -> Result[PipelineSpokeSpec]"""
    def parse(spec: PipelineSpokeSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: PipelineSpokeSpec -> Result[PipelineSpokeSpec]"""
    def validate(spec: PipelineSpokeSpec) -> Result:
        return validate_spec_instance(spec)
    return validate


def _make_fix(base_url: str = "", ask_fn=None):
    """G'_pipeline: (spec, error, ctx) -> spec | None"""
    def fix(spec: PipelineSpokeSpec, error: str, ctx: GenerationContext):
        # Simple fix: return None to trigger a fresh generation
        logger.warning("Ouroboros fix not implemented for pipeline spokes")
        return None
    return fix


def _make_emit(output_dir: Path):
    """IO: write spoke to output directory."""
    def emit(
        spec: PipelineSpokeSpec,
        artifact: PipelineSpokeSpec,
        rounds: int,
        fixes: int,
        version_hint: int,
        prompt_or_claim,
    ) -> Result:
        try:
            spoke_path = emit_spoke(spec, output_dir, version=version_hint)
            report = GenerationReport(
                version=version_hint,
                rounds=rounds,
                ouroboros_fixes=fixes,
                outcome="success",
                claim=spec.description,
                user_prompt=prompt_or_claim if isinstance(prompt_or_claim, str) else None,
            )
            report_path = spoke_path.with_suffix(".report.json")
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            return Ok(spoke_path)
        except Exception as exc:
            return Err(f"Emit error: {exc}")
    return emit


def emit_spoke(
    spec: PipelineSpokeSpec,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write a PipelineSpokeSpec to disk as a .py file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in spec.name
    ).strip().replace(" ", "_").lower()

    if version > 0:
        filename = f"{safe_name}_v{version}.py"
    else:
        filename = f"{safe_name}.py"

    out_path = output_dir / filename

    # Generate the spoke source code
    source = _generate_spoke_source(spec)

    out_path.write_text(source)
    logger.info("Spoke written to %s", out_path)
    return out_path


def _generate_spoke_source(spec: PipelineSpokeSpec) -> str:
    """Generate the Python source code for the spoke."""
    stages_code = []
    for stage_name, stage_config in spec.stages.items():
        # Escape the prompt for Python string
        prompt_escaped = stage_config.prompt.replace('\\', '\\\\').replace('"', '\\"')
        stages_code.append(f'        "{stage_name}": StageConfig(model="{stage_config.model}", prompt="""{prompt_escaped}"""),')

    stages_block = "\n".join(stages_code)

    source = f'''#!/usr/bin/env python3
"""
Pipeline spoke: {spec.name}

{spec.description}

This spoke implements a multi-stage pipeline where each stage
uses a specific model and passes its output to the next stage.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add four to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory

# Pipeline configuration
PIPELINE = {{
    "name": "{spec.name}",
    "description": "{spec.description}",
    "stages": {{
{stages_block}
    }},
    "output_dir": "{spec.output_dir}",
}}


def slugify(text: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    return text


def execute_stage(stage_name: str, stage_config: dict[str, Any], input_text: str) -> str:
    """Execute a single pipeline stage."""
    model_name = stage_config["model"]
    model_id = model_name.split(":")[0] if ":" in model_name else model_name
    MODEL_ID = os.getenv("FIVE_MODEL", model_id)
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))

    def invoke(messages):
        result = litellm_invoke(
            model=f"openai/{MODEL_ID}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )(messages)
        return result

    def parse(raw):
        return regex_parse()(raw)

    def validate(action):
        return local_env()(action)

    def emit(messages, outcome):
        return Path("trajectory.json")

    system_prompt = stage_config["prompt"]
    prompt = input_text

    path = run(
        G=invoke,
        V1=parse,
        V2=validate,
        emit=emit,
        system=system_prompt,
        prompt=prompt,
        max_steps=10,
    )

    # Read the trajectory to get the output
    trajectory = Path("trajectory.json")
    if trajectory.exists():
        import json
        data = json.loads(trajectory.read_text())
        # Get the final assistant message
        for msg in reversed(data.get("messages", [])):
            if msg.get("role") == "assistant":
                return msg.get("content", "")

    return "Stage output not found"


def main():
    parser = argparse.ArgumentParser(description=PIPELINE["description"])
    parser.add_argument("--topic", required=True, help="Pipeline topic/input")
    parser.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    topic = args.topic
    slug = slugify(topic)
    output_dir = Path(PIPELINE["output_dir"]) / slug

    print(f"Topic: {topic}")
    print(f"Slug: {slug}")
    print(f"Output dir: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the topic
    (output_dir / "topic.txt").write_text(topic)

    # Execute pipeline stages
    current_input = topic
    stage_names = list(PIPELINE["stages"].keys())

    for i, stage_name in enumerate(stage_names):
        stage_config = PIPELINE["stages"][stage_name]
        output_file = output_dir / f"{stage_name}.md"

        print(f"Stage {i+1}/{len(stage_names)}: {stage_name}")
        print("-" * 40)

        t0 = time.time()
        output = execute_stage(stage_name, stage_config, current_input)
        elapsed = time.time() - t0

        output_file.write_text(output)
        print(f"  Output: {output_file} ({elapsed:.1f}s)")
        print()

        current_input = output

    # Save final output
    final_file = output_dir / "essay.md"
    final_file.write_text(current_input)

    print("=" * 40)
    print("COMPLETE")
    print("=" * 40)
    print(f"Topic: {topic}")
    print()
    for stage_name in stage_names:
        file_path = output_dir / f"{stage_name}.md"
        print(f"{stage_name.capitalize()}: {file_path}")
    print()


if __name__ == "__main__":
    main()
'''

    return source


def run(
    prompt: str | None = None,
    *,
    base_url: str = "",
    output: str | Path | None = None,
    max_rounds: int = 3,
    max_fixes: int = 3,
    verbose: bool = False,
    dry_run: bool = False,
) -> Result:
    """Programmatic entry point."""
    if not prompt:
        return Err("prompt is required")
    out = Path(output) if output else Path.cwd() / "spokes"

    logger.info("Building pipeline spoke generation context...")
    ctx = build_pipeline_context()
    ctx = ctx.with_prompt(prompt or "")

    if not base_url:
        base_url = os.getenv("FIVE_BASE_URL", base_url)

    if dry_run:
        from four.generators._invoke import build_system_prompt, build_user_message
        system = build_system_prompt(
            ctx,
            _TYPES_SOURCE,
            role=(
                "You are an expert spoke developer. Generate a pipeline spoke."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing PipelineSpokeSpec."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write a PipelineSpokeSpec(...) expression.",
                "No markdown fencing.",
            ),
        )
        print("=" * 80)
        print("SYSTEM PROMPT")
        print("=" * 80)
        print(system)
        print()
        print("=" * 80)
        print("USER MESSAGE")
        print("=" * 80)
        print(user)
        return Ok(Path("<dry-run>"))

    match generation_loop(
        ctx,
        invoke=_make_invoke(base_url),
        parse=_make_parse(),
        validate=_make_validate(),
        fix=_make_fix(base_url),
        emit=_make_emit(out),
        max_rounds=max_rounds,
        max_fixes=max_fixes,
    ):
        case Ok(path):
            logger.info("Generated: %s", path)
            return Ok(path)
        case Err(e):
            logger.error("Generation failed: %s", e)
            return Err(e)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline spoke generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="What pipeline spoke to generate"
    )
    parser.add_argument(
        "--base-url", type=str, default="",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: ./spokes)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="Max rounds (default: 3)",
    )
    parser.add_argument(
        "--max-fixes", type=int, default=3,
        help="Max fixes per round (default: 3)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling the model",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Interactive REPL mode",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.live:
        from four.generators._loop import repl_loop

        ctx = build_pipeline_context()
        return result_to_exit(repl_loop(
            ctx,
            invoke=_make_invoke(args.base_url),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(args.base_url),
            emit=_make_emit(Path(args.output_dir or "spokes")),
            max_rounds=args.max_rounds,
            max_fixes=args.max_fixes,
            banner="Pipeline Spoke Generator -- interactive mode",
        ))

    result = run(
        args.prompt,
        base_url=args.base_url,
        output=args.output_dir,
        max_rounds=args.max_rounds,
        max_fixes=args.max_fixes,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    match result:
        case Ok(path):
            print(f"Spoke generated: {path}")
        case Err(e):
            print(f"Generation failed: {e}", file=sys.stderr)

    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
