#!/usr/bin/env python3
"""Four-agent generator -- generates four-style functional algebra agents.

G_four    : Query -> FourAgentSpec
V_four    : syntax check + import check + structure check
G'_four   : (FourAgentSpec, Error) -> AgentPatch -> FourAgentSpec

Usage:
    python -m four.generators.four_agent.generate \\
        --prompt "Create a bash agent with chat variant" \\
        --output-dir ./agents

    python -m four.generators.four_agent.generate \\
        --prompt "Create a toolcall agent for file operations" --dry-run

    # Interactive REPL
    python -m four.generators.four_agent.generate --live

    # Refine existing agent
    python -m four.generators.four_agent.generate \\
        --refine agents/bash_agent.py \\
        --claim "Add toolcall support"
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

from ._types import FourAgentSpec, validate_spec_instance
from ._runtime import (
    emit_agent,
    invoke_model,
    load_agent,
    ouroboros_agent,
    validate_agent,
)
from ._context import build_four_agent_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


def _make_invoke(base_url: str = "", ask_fn=None):
    """G_four: Context -> Result[FourAgentSpec]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, base_url, ask_fn)
    return invoke


def _make_parse():
    """V1: FourAgentSpec -> Result[FourAgentSpec]"""
    def parse(spec: FourAgentSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: FourAgentSpec -> Result[FourAgentSpec]"""
    def validate(spec: FourAgentSpec) -> Result:
        return validate_agent(spec)
    return validate


def _make_fix(base_url: str = "", ask_fn=None):
    """G'_four: (spec, error, ctx) -> spec | None"""
    def fix(spec: FourAgentSpec, error: str, ctx: GenerationContext):
        return ouroboros_agent(spec, error, ctx, base_url, ask_fn)
    return fix


def _make_emit(output_dir: Path):
    """IO: write agent to output directory."""
    def emit(
        spec: FourAgentSpec,
        artifact: FourAgentSpec,
        rounds: int,
        fixes: int,
        version_hint: int,
        prompt_or_claim,
    ) -> Result:
        try:
            agent_path = emit_agent(spec, output_dir, version=version_hint)
            report = GenerationReport(
                version=version_hint,
                rounds=rounds,
                ouroboros_fixes=fixes,
                outcome="success",
                claim=spec.title,
                user_prompt=prompt_or_claim if isinstance(prompt_or_claim, str) else None,
            )
            report_path = agent_path.with_suffix(".report.json")
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            return Ok(agent_path)
        except Exception as exc:
            return Err(f"Emit error: {exc}")
    return emit


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run(
    prompt: str | None = None,
    *,
    base_url: str = "",
    output: str | Path | None = None,
    max_rounds: int = 3,
    max_fixes: int = 3,
    verbose: bool = False,
    dry_run: bool = False,
    refine: tuple[str, str] | None = None,
) -> Result:
    """Programmatic entry point. No argparse, no sys.argv."""
    if not prompt and not refine:
        return Err("prompt or refine is required")
    out = Path(output) if output else Path.cwd() / "agents"

    logger.info("Building four-agent generation context...")
    ctx = build_four_agent_context(prompt)

    if refine:
        artifact_path, claim = refine
        # Load raw file content for refinement context
        raw_content = Path(artifact_path).read_text()
        ctx = ctx.with_domain(DomainSection(
            heading="Existing Agent (modify according to claim)",
            content=raw_content,
        ))
        ctx = ctx.with_prompt(f"{claim}\n\nKeep the rest of the agent structure the same, only modify what's necessary to implement this change.")
    else:
        ctx = ctx.with_prompt(prompt or "")

    # Use FIVE_BASE_URL if base_url is not explicitly set
    if not base_url:
        base_url = os.getenv("FIVE_BASE_URL", base_url)

    if dry_run:
        from four.generators._invoke import build_system_prompt, build_user_message
        system = build_system_prompt(
            ctx,
            (Path(__file__).parent / "_types.py").read_text(),
            role=(
                "You are an expert agent developer specializing in the four-framework, "
                "a four-function algebra for building LLM agents."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing FourAgentSpec."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write a FourAgentSpec(...) expression.",
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
        description="Four-agent generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, help="What agent to generate"
    )
    parser.add_argument(
        "--base-url", type=str, default="",
        help="OpenAI-compatible API base URL (default: $LLM_BASE_URL or localhost:8080/v1)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: ./agents)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="Max wholesale rounds (default: 3)",
    )
    parser.add_argument(
        "--max-fixes", type=int, default=3,
        help="Max ouroboros fixes per round (default: 3)",
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
    parser.add_argument(
        "--refine", type=str, default=None,
        help="Path to existing agent to refine",
    )
    parser.add_argument(
        "--claim", type=str, default=None,
        help="Refinement claim (used with --refine)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.live:
        from four.generators._loop import repl_loop

        ctx = build_four_agent_context()
        return result_to_exit(repl_loop(
            ctx,
            invoke=_make_invoke(args.base_url),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(args.base_url),
            emit=_make_emit(Path(args.output_dir or "agents")),
            max_rounds=args.max_rounds,
            max_fixes=args.max_fixes,
            load_artifact=load_agent,
            banner="Four-Agent Generator -- interactive mode",
        ))

    refine = None
    if args.refine:
        if not args.claim:
            print("Error: --claim is required with --refine", file=sys.stderr)
            return 1
        refine = (args.refine, args.claim)

    result = run(
        args.prompt,
        base_url=args.base_url,
        output=args.output_dir,
        max_rounds=args.max_rounds,
        max_fixes=args.max_fixes,
        verbose=args.verbose,
        dry_run=args.dry_run,
        refine=refine,
    )

    match result:
        case Ok(path):
            print(f"Agent generated: {path}")
        case Err(e):
            print(f"Generation failed: {e}", file=sys.stderr)

    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
