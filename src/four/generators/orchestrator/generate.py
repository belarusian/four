#!/usr/bin/env python3
"""Orchestrator generator — generates a new orchestrator using idea-forge templates.

Usage:
    python -m four.generators.orchestrator.generate \\
        --prompt "Create an orchestrator for email-driven GitHub workflow" \\
        --output-dir ./orchestrators

    python -m four.generators.orchestrator.generate --live
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

from ._types import OrchestratorSpec, ManifestEntry
from ._validation import validate_spec_instance
from ._runtime import invoke_model, _TYPES_SOURCE
from ._context import build_orchestrator_context
from four.generators._invoke import build_system_prompt, build_user_message

logger = logging.getLogger(__name__)


def _make_invoke(base_url: str = "", ask_fn=None):
    """G_orchestrator: Context -> Result[OrchestratorSpec]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, base_url, ask_fn)
    return invoke


def _make_parse():
    """V1: OrchestratorSpec -> Result[OrchestratorSpec]"""
    def parse(spec: OrchestratorSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: OrchestratorSpec -> Result[OrchestratorSpec]"""
    def validate(spec: OrchestratorSpec) -> Result:
        return validate_spec_instance(spec)
    return validate


def _make_fix(base_url: str = "", ask_fn=None):
    """G'_orchestrator: (spec, error, ctx) -> spec | None"""
    def fix(spec: OrchestratorSpec, error: str, ctx: GenerationContext):
        logger.warning("Ouroboros fix not implemented for orchestrators")
        return None
    return fix


def _make_emit(output_dir: Path):
    """IO: write orchestrator directory to output directory."""
    def emit(
        spec: OrchestratorSpec,
        artifact: OrchestratorSpec,
        rounds: int,
        fixes: int,
        version_hint: int,
        prompt_or_claim,
    ) -> Result:
        try:
            orch_path = emit_orchestrator(spec, output_dir, version=version_hint)
            report = GenerationReport(
                version=version_hint,
                rounds=rounds,
                ouroboros_fixes=fixes,
                outcome="success",
                claim=spec.description,
                user_prompt=prompt_or_claim if isinstance(prompt_or_claim, str) else None,
            )
            report_path = orch_path / "report.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            return Ok(orch_path)
        except Exception as exc:
            return Err(f"Emit error: {exc}")
    return emit


def emit_orchestrator(
    spec: OrchestratorSpec,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write an OrchestratorSpec to disk as an orchestrator directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in spec.name
    ).strip().replace(" ", "_").lower()

    if version > 0:
        dir_name = f"{safe_name}_v{version}"
    else:
        dir_name = safe_name

    orch_dir = output_dir / dir_name
    orch_dir.mkdir(exist_ok=True)

    # Create project structure using idea-forge templates
    # (In practice, this would call idea-forge or generate the structure)

    # Create manifest
    manifest_path = orch_dir / "manifests" / "projects.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write manifest header
    lines = ["# repo\ttitle\tdescription\ttemplates\tstarter_pack\tsummary"]
    for entry in spec.manifest_entries:
        templates_csv = ",".join(entry.templates)
        lines.append(f"{entry.repo}\t{entry.title}\t{entry.description}\t{templates_csv}\t\t{entry.description}")
    
    manifest_path.write_text("\n".join(lines) + "\n")
    logger.info("Manifest written to %s", manifest_path)

    # Create README
    (orch_dir / "README.md").write_text(f"# {spec.name}\n\n{spec.description}\n")
    
    return orch_dir


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
    out = Path(output) if output else Path.cwd() / "orchestrators"

    logger.info("Building orchestrator generation context...")
    ctx = build_orchestrator_context()
    ctx = ctx.with_prompt(prompt or "")

    if not base_url:
        base_url = os.getenv("FIVE_BASE_URL", base_url)

    if dry_run:
        system = build_system_prompt(
            ctx,
            _TYPES_SOURCE,
            role=(
                "You are an expert orchestrator developer. "
                "Generate a complete orchestrator system for distributed LLM agents."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing OrchestratorSpec."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write an OrchestratorSpec(...) expression.",
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
        description="Orchestrator generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="What orchestrator to generate"
    )
    parser.add_argument(
        "--base-url", type=str, default="",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: ./orchestrators)",
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

        ctx = build_orchestrator_context()
        return result_to_exit(repl_loop(
            ctx,
            invoke=_make_invoke(args.base_url),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(args.base_url),
            emit=_make_emit(Path(args.output_dir or "orchestrators")),
            max_rounds=args.max_rounds,
            max_fixes=args.max_fixes,
            banner="Orchestrator Generator -- interactive mode",
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
            print(f"Orchestrator generated: {path}")
        case Err(e):
            print(f"Generation failed: {e}", file=sys.stderr)

    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
