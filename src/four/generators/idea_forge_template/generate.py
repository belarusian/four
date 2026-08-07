"""Idea-forge template generator.

Usage:
    python -m four.generators.idea_forge_template.generate \\
        --prompt "Create a Python research template" \\
        --output-dir ./templates

    python -m four.generators.idea_forge_template.generate --live
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

from ._types import IdeaForgeTemplate
from ._runtime import invoke_model, _TYPES_SOURCE
from ._validation import validate_spec_instance
from four.generators._invoke import build_system_prompt, build_user_message
from ._context import build_idea_forge_context

logger = logging.getLogger(__name__)


def _make_invoke(base_url: str = "", ask_fn=None):
    """G_template: Context -> Result[IdeaForgeTemplate]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, base_url, ask_fn)
    return invoke


def _make_parse():
    """V1: IdeaForgeTemplate -> Result[IdeaForgeTemplate]"""
    def parse(spec: IdeaForgeTemplate) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: IdeaForgeTemplate -> Result[IdeaForgeTemplate]"""
    def validate(spec: IdeaForgeTemplate) -> Result:
        return validate_spec_instance(spec)
    return validate


def _make_fix(base_url: str = "", ask_fn=None):
    """G'_template: (spec, error, ctx) -> spec | None"""
    def fix(spec: IdeaForgeTemplate, error: str, ctx: GenerationContext):
        logger.warning("Ouroboros fix not implemented for idea-forge templates")
        return None
    return fix


def _make_emit(output_dir: Path):
    """IO: write template directory to output directory."""
    def emit(
        spec: IdeaForgeTemplate,
        artifact: IdeaForgeTemplate,
        rounds: int,
        fixes: int,
        version_hint: int,
        prompt_or_claim,
    ) -> Result:
        try:
            template_path = emit_template(spec, output_dir, version=version_hint)
            report = GenerationReport(
                version=version_hint,
                rounds=rounds,
                ouroboros_fixes=fixes,
                outcome="success",
                claim=spec.description,
                user_prompt=prompt_or_claim if isinstance(prompt_or_claim, str) else None,
            )
            report_path = template_path / "report.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            return Ok(template_path)
        except Exception as exc:
            return Err(f"Emit error: {exc}")
    return emit


def emit_template(
    spec: IdeaForgeTemplate,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write an IdeaForgeTemplate to disk as a template directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in spec.name
    ).strip().replace(" ", "_").lower()

    if version > 0:
        dir_name = f"{safe_name}_v{version}"
    else:
        dir_name = safe_name

    template_dir = output_dir / dir_name
    template_dir.mkdir(exist_ok=True)

    # Write README.md
    (template_dir / "README.md").write_text(spec.readme)
    logger.info("README.md written to %s", template_dir / "README.md")

    # Write template.env
    (template_dir / "template.env").write_text(spec.template_env)
    logger.info("template.env written to %s", template_dir / "template.env")

    return template_dir


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
    out = Path(output) if output else Path.cwd() / "templates"

    logger.info("Building idea-forge template generation context...")
    ctx = build_idea_forge_context()
    ctx = ctx.with_prompt(prompt or "")

    if not base_url:
        base_url = os.getenv("FIVE_BASE_URL", base_url)

    if dry_run:
        system = build_system_prompt(
            ctx,
            _TYPES_SOURCE,
            role=(
                "You are an expert repository template developer for idea-forge."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing IdeaForgeTemplate."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write an IdeaForgeTemplate(...) expression.",
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
        description="Idea-forge template generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="What template to generate"
    )
    parser.add_argument(
        "--base-url", type=str, default="",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: ./templates)",
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

        ctx = build_idea_forge_context()
        return result_to_exit(repl_loop(
            ctx,
            invoke=_make_invoke(args.base_url),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(args.base_url),
            emit=_make_emit(Path(args.output_dir or "templates")),
            max_rounds=args.max_rounds,
            max_fixes=args.max_fixes,
            banner="Idea-Forge Template Generator -- interactive mode",
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
            print(f"Template generated: {path}")
        case Err(e):
            print(f"Generation failed: {e}", file=sys.stderr)

    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
