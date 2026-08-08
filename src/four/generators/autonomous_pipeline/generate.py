"""Autonomous pipeline generator.

Usage:
    python -m four.generators.autonomous_pipeline.generate \\
        --prompt "Create an autonomous pipeline to implement feature X" \\
        --output-dir ./spokes

    python -m four.generators.autonomous_pipeline.generate --live
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

from ._types import AutonomousPipelineSpec
from ._context import build_autonomous_pipeline_context
from ._runtime import _TYPES_SOURCE, invoke_model
from ._validation import validate_spec_instance

logger = logging.getLogger(__name__)


def _make_invoke(base_url: str = "", ask_fn=None):
    """G_autonomous: Context -> Result[AutonomousPipelineSpec]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, base_url, ask_fn)
    return invoke


def _make_parse():
    """V1: AutonomousPipelineSpec -> Result[AutonomousPipelineSpec]"""
    def parse(spec: AutonomousPipelineSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: AutonomousPipelineSpec -> Result[AutonomousPipelineSpec]"""
    def validate(spec: AutonomousPipelineSpec) -> Result:
        return validate_spec_instance(spec)
    return validate


def _make_fix(base_url: str = "", ask_fn=None):
    """G'_autonomous: (spec, error, ctx) -> spec | None"""
    def fix(spec: AutonomousPipelineSpec, error: str, ctx: GenerationContext):
        # Simple fix: return None to trigger a fresh generation
        logger.warning("Ouroboros fix not implemented for autonomous pipelines")
        return None
    return fix


def _make_emit(output_dir: Path):
    """IO: write spoke to output directory."""
    def emit(
        spec: AutonomousPipelineSpec,
        artifact: AutonomousPipelineSpec,
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
    spec: AutonomousPipelineSpec,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write an AutonomousPipelineSpec to disk as a .py file."""
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

    source = _generate_spoke_source(spec)
    out_path.write_text(source)
    logger.info("Spoke written to %s", out_path)
    return out_path


def _generate_spoke_source(spec: AutonomousPipelineSpec) -> str:
    """Generate the Python source code for the autonomous pipeline spoke."""
    stages_block = "\n".join(
        f'    - {name}: {stage.description} (max {stage.max_turns} turns, validate: {stage.validation_command})'
        for name, stage in spec.stages.items()
    )

    source = f'''#!/usr/bin/env python3
"""
Autonomous pipeline: {spec.name}

{spec.description}

This spoke implements an autonomous development loop using four.core.run().
The agent uses bash to write code, run git, execute tests, and iterate.

Stages:
{stages_block}
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory


def main():
    parser = argparse.ArgumentParser(description="{spec.description}")
    parser.add_argument("--goal", required=True, help="The goal to achieve")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps")
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "granite4.1:8b")
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "65536"))

    step_num = [0]

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = litellm_invoke(
            model=f"openai/{{MODEL_ID}}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\\n", " ")
            print(f"  [G step {{step_num[0]}}] ({{elapsed:.1f}}s) -> {{preview}}...")
        else:
            print(f"  [G step {{step_num[0]}}] ({{elapsed:.1f}}s) ERR: {{result.error[:100]}}")
        return result

    system = """You are an autonomous development agent.

Goal: Implement the user's request by writing code, committing, and pushing.

Stages to follow:
{stages_block}

Use bash commands to:
- Write/modify files with cat, echo, or python -c
- Run git add, git commit, git push
- Execute tests, linters, builds
- Call other LLMs via curl if needed

Each step, output a bash command in a code block. The system will execute it and show you the result. Continue until the goal is achieved.

When done, output: DONE
"""

    path = run(
        G=debug_g,
        V1=regex_parse(),
        V2=local_env(),
        emit=save_trajectory(),
        system=system,
        prompt=args.goal,
        max_steps=args.max_steps,
    )
    print(f"Trajectory saved to: {{path}}")
    print(f"Total G calls: {{step_num[0]}}")


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

    logger.info("Building autonomous pipeline generation context...")
    ctx = build_autonomous_pipeline_context()
    ctx = ctx.with_prompt(prompt or "")

    if not base_url:
        base_url = os.getenv("FIVE_BASE_URL", base_url)

    if dry_run:
        from four.generators._invoke import build_system_prompt, build_user_message
        system = build_system_prompt(
            ctx,
            _TYPES_SOURCE,
            role=(
                "You are an expert at generating autonomous pipeline spokes. "
                "Generate a Python script that implements a long-running autonomous development loop."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing AutonomousPipelineSpec."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write an AutonomousPipelineSpec(...) expression.",
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
        description="Autonomous pipeline generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="What autonomous pipeline to generate"
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

        ctx = build_autonomous_pipeline_context()
        return result_to_exit(repl_loop(
            ctx,
            invoke=_make_invoke(args.base_url),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(args.base_url),
            emit=_make_emit(Path(args.output_dir or "spokes")),
            max_rounds=args.max_rounds,
            max_fixes=args.max_fixes,
            banner="Autonomous Pipeline Generator -- interactive mode",
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
