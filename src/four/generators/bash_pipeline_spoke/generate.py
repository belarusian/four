"""Bash pipeline spoke generator.

Usage:
    python -m four.generators.bash_pipeline_spoke.generate \\
        --prompt "Create an essay pipeline spoke" \\
        --output-dir ./spokes

    python -m four.generators.bash_pipeline_spoke.generate --live
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

from ._types import BashPipelineSpec
from ._context import build_bash_pipeline_context
from ._runtime import (
    _TYPES_SOURCE,
    invoke_model,
)
from ._validation import validate_spec_instance

logger = logging.getLogger(__name__)


def _make_invoke(base_url: str = "", ask_fn=None):
    """G_bash: Context -> Result[BashPipelineSpec]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, base_url, ask_fn)
    return invoke


def _make_parse():
    """V1: BashPipelineSpec -> Result[BashPipelineSpec]"""
    def parse(spec: BashPipelineSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: BashPipelineSpec -> Result[BashPipelineSpec]"""
    def validate(spec: BashPipelineSpec) -> Result:
        return validate_spec_instance(spec)
    return validate


def _make_fix(base_url: str = "", ask_fn=None):
    """G'_bash: (spec, error, ctx) -> spec | None"""
    def fix(spec: BashPipelineSpec, error: str, ctx: GenerationContext) -> BashPipelineSpec | None:
        # Simple fix: return None to trigger a fresh generation with error feedback
        logger.warning("Ouroboros fix not implemented for bash pipeline spokes")
        return None
    return fix


def _make_emit(output_dir: Path):
    """IO: write spoke to output directory."""
    def emit(
        spec: BashPipelineSpec,
        artifact: BashPipelineSpec,
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
    spec: BashPipelineSpec,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write a BashPipelineSpec to disk as a .sh file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in spec.name
    ).strip().replace(" ", "_").lower()

    if version > 0:
        filename = f"{safe_name}_v{version}.sh"
    else:
        filename = f"{safe_name}.sh"

    out_path = output_dir / filename

    # Use the source from spec (full bash script from model)
    source = spec.source if spec.source else "#!/usr/bin/env bash\n# No source provided\n"

    out_path.write_text(source)
    logger.info("Spoke written to %s", out_path)
    return out_path


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

    logger.info("Building bash pipeline spoke generation context...")
    ctx = build_bash_pipeline_context()
    ctx = ctx.with_prompt(prompt or "")

    if not base_url:
        base_url = os.getenv("FIVE_BASE_URL", base_url)

    if dry_run:
        from four.generators._invoke import build_system_prompt, build_user_message
        system = build_system_prompt(
            ctx,
            _TYPES_SOURCE,
            role=(
                "You are an expert at generating bash pipeline spokes for the four framework. "
                "Generate a bash script that implements a multi-stage pipeline."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing BashPipelineSpec."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write a BashPipelineSpec(...) expression.",
                "Include the COMPLETE bash source code in the source field.",
                "Use dict syntax for stages: stages={\"stage1\": {\"model\": \"...\", \"prompt\": \"...\"}}",
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
        description="Bash pipeline spoke generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="What bash pipeline spoke to generate"
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

        ctx = build_bash_pipeline_context()
        return result_to_exit(repl_loop(
            ctx,
            invoke=_make_invoke(args.base_url),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(args.base_url),
            emit=_make_emit(Path(args.output_dir or "spokes")),
            max_rounds=args.max_rounds,
            max_fixes=args.max_fixes,
            banner="Bash Pipeline Spoke Generator -- interactive mode",
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
