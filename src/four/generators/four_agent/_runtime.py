"""IO boundaries for the four-agent generator.

G_four: invoke model, return FourAgentSpec
V_four: structural -> AST -> exec (dry-run validation)
G'_four: patch-based ouroboros
IO: emit .py file
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from four.generators._types import (
    AskFn,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from four.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)
from four.generators._validation import (
    validate_python_sources,
)

from . import _types
from ._types import AgentPatch, Edit, FourAgentSpec, apply_patch, validate_spec_instance

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


# ---------------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------------


def invoke_model(
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return a FourAgentSpec. G_four."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert agent developer specializing in the four-framework, "
            "a four-function algebra for building LLM agents. You generate "
            "well-structured, executable Python scripts that implement agents "
            "using the G -> V1 -> V2* -> emit pattern."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing FourAgentSpec.\n"
            "See the docstring for the exact response format.\n"
            "The 'source' field must contain the COMPLETE Python source code "
            "for the agent, including all imports, function definitions, and "
            "the main() entry point."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write a FourAgentSpec(...) expression.",
            "Put the COMPLETE agent source code as a TRIPLE-QUOTED string in the source field.",
            "Triple quotes avoid escaping issues with newlines and special characters.",
            "No markdown fencing.",
            "",
            "Guidelines:",
            "- Include #!/usr/bin/env python3 shebang",
            "- Import from four.core, four.chat_model, four.parse, four.env",
            "- Use argparse for --prompt and --endpoint arguments",
            "- Wrap the transport in a debug_g function with timing and logging",
            "- Define V1 (parse), V2 (validate), and emit functions",
            "- Call run(G=debug_g, V1=v1, V2=v2, emit=emit, ...) at the end",
            "- Keep the code under 100 lines",
            "- All code must be syntactically valid Python",
            "- Use triple quotes: source=\"\"\"...\"\"\"",
        ),
    )

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    # Parse response
    try:
        spec = _parse_response(raw_text)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


def _parse_response(raw_text: str) -> FourAgentSpec:
    """Parse the model's Python constructor response into a spec."""
    import re

    # First try: extract from FourAgentSpec(...) constructor
    start_marker = "FourAgentSpec("
    start_idx = raw_text.find(start_marker)

    if start_idx >= 0:
        # Extract the constructor body by balancing parens
        depth = 0
        end_idx = start_idx
        for i in range(start_idx, len(raw_text)):
            ch = raw_text[i]
            # Skip chars inside strings
            if i > start_idx and (raw_text[i-1:i] == '\\'):
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        constructor_body = raw_text[start_idx + len(start_marker):end_idx]

        # Extract simple fields
        title = _extract_field(constructor_body, "title") or "Four Agent"
        description = _extract_field(constructor_body, "description") or "Agent"
        variant = _extract_field(constructor_body, "variant") or "chat"
        system_prompt = _extract_field(constructor_body, "system_prompt") or ""

        # Extract source - everything after source= until closing )
        source = _extract_source(constructor_body)
    else:
        # Fallback: look for python code block
        title = "Four Agent"
        description = "Agent"
        variant = "chat"
        system_prompt = ""
        source = _extract_code_block(raw_text)

    if not source.strip():
        raise ValueError("source field is empty or could not be extracted")

    return FourAgentSpec(
        title=title,
        description=description,
        variant=variant,
        system_prompt=system_prompt,
        source=source,
    )


def _extract_field(body: str, field_name: str) -> str | None:
    """Extract a simple string field from constructor body."""
    m = re.search(rf"{field_name}\s*=\s*['\"](.*?)['\"]", body)
    if m:
        return m.group(1)
    return None


def _extract_source(body: str) -> str:
    """Extract source field, handling triple quotes and escaped content."""
    # Try triple-quoted string first
    for quote in ('"""', "'''"):
        pattern = rf"source\s*=\s*{quote}(.*?){quote}"
        m = re.search(pattern, body, re.DOTALL)
        if m:
            return m.group(1)

    # Try regular quoted string
    m = re.search(r"source\s*=\s*['\"]", body)
    if m:
        quote_char = m.group(0)[-1]
        source_start = m.end()

        # Find closing quote, handling escapes
        i = source_start
        while i < len(body):
            ch = body[i]
            if ch == '\\' and i + 1 < len(body):
                # Handle escape sequences
                next_ch = body[i+1]
                if next_ch == 'n':
                    # This might be \n in the model output
                    # Keep scanning to find the actual closing quote
                    pass
                i += 2
                continue
            if ch == quote_char:
                rest = body[i+1:].lstrip()
                if rest.startswith(')') or rest.startswith('),') or rest == '':
                    extracted = body[source_start:i]
                    # Try to decode unicode escape sequences
                    try:
                        return extracted.encode().decode('unicode_escape', errors='replace')
                    except:
                        return extracted
            i += 1

    # Last resort: everything after source= until the end
    m = re.search(r"source\s*=\s*['\"]?", body)
    if m:
        extracted = body[m.end():].rstrip(')"\'')
        try:
            return extracted.encode().decode('unicode_escape', errors='replace')
        except:
            return extracted

    return ""


def _extract_code_block(text: str) -> str:
    """Extract Python code from markdown code block as fallback."""
    import re
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


# ---------------------------------------------------------------------------
# Validation pipeline -- V_four
# ---------------------------------------------------------------------------


def validate_agent(spec: FourAgentSpec) -> Result[FourAgentSpec, str]:
    """Full V_four pipeline. Cheapest first:

    1. Structural: validate_spec_instance
    2. Semantic: ast.parse() the source
    3. Import check: verify required imports are present
    """
    # 1. Structural
    match validate_spec_instance(spec):
        case Err(e):
            return Err(e)

    # 2. Semantic: syntax check
    match validate_python_sources((spec.source,), label="agent source"):
        case Err(e):
            return Err(e)

    logger.info("Agent source parses cleanly")

    # 3. Import check
    match _validate_imports(spec):
        case Err(e):
            return Err(e)

    # 4. Structure check: must have main() and run() call
    match _validate_structure(spec):
        case Err(e):
            return Err(e)

    return Ok(spec)


def _validate_imports(spec: FourAgentSpec) -> Result[None, str]:
    """Check that required imports are present."""
    source = spec.source

    required_imports = ["from four.core import"]
    for imp in required_imports:
        if imp not in source:
            return Err(
                f"Agent source is missing required import: '{imp}'. "
                f"Add 'from four.core import run, Ok, Err' at the top."
            )

    variant_imports = {
        "chat": "from four.chat_model import litellm_invoke",
        "toolcall": "from four.chat_model import litellm_toolcall_invoke",
        "responses": "from four.response_model import http_response_invoke",
    }

    variant_parse = {
        "chat": "from four.parse import regex_parse",
        "toolcall": "from four.parse import toolcall_parse",
        "responses": "from four.parse import toolcall_parse",
    }

    needed_import = variant_imports.get(spec.variant)
    if needed_import and needed_import not in source:
        return Err(
            f"Agent variant '{spec.variant}' requires import: "
            f"'{needed_import}'"
        )

    needed_parse = variant_parse.get(spec.variant)
    if needed_parse and needed_parse not in source:
        return Err(
            f"Agent variant '{spec.variant}' requires import: "
            f"'{variant_parse[spec.variant]}'"
        )

    if "from four.env import local_env" not in source:
        return Err(
            "Agent source is missing 'from four.env import local_env'. "
            "Add this import for the V2 (validate) function."
        )

    return Ok(None)


def _validate_structure(spec: FourAgentSpec) -> Result[None, str]:
    """Check that the agent has proper structure."""
    import ast

    source = spec.source

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Ok(None)  # Already checked by AST validation

    # Check for main() function
    has_main = False
    has_run_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            has_main = True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "run":
                has_run_call = True

    if not has_main:
        return Err(
            "Agent source must have a main() function. "
            "Wrap the agent logic in a main() function."
        )

    if not has_run_call:
        return Err(
            "Agent source must call run(G=..., V1=..., V2=..., emit=..., ...). "
            "Add the four.core.run() call to execute the agent loop."
        )

    return Ok(None)


# ---------------------------------------------------------------------------
# Ouroboros -- patch-based
# ---------------------------------------------------------------------------


def ouroboros_agent(
    spec: FourAgentSpec,
    error: str,
    ctx: GenerationContext,
    base_url: str = "",
    ask_fn: AskFn | None = None,
) -> FourAgentSpec | None:
    """G'_four: model sees prior agent + error, returns an AgentPatch."""
    fn = resolve_ask_fn(base_url, ask_fn)

    system_parts = [
        "You are correcting a four-style agent you previously generated.",
        f"Agent: {spec.title}",
        f"Variant: {spec.variant}",
        "",
        "You will receive your prior agent source code and an error.",
        "Write an AgentPatch(...) expression with targeted edits.",
        "Each edit targets an exact substring with search-and-replace.",
        "Only include the parts that need to change.",
        "",
        "AgentPatch format:",
        "AgentPatch(",
        "    edits=(",
        '        Edit(old="exact text to find", new="replacement text"),',
        "    ),",
        ")",
        "",
        "Rules:",
        "- 'old' must be an EXACT substring of the current source.",
        "- 'old' should include enough surrounding context to be unique.",
        "- Fix only what's needed to resolve the error.",
        "- No markdown fencing, no explanation.",
    ]

    user_parts = [
        f"# Your agent: {spec.title} ({spec.variant} variant)",
        "",
        "```python",
        spec.source,
        "```",
        "",
        "# Error",
        "",
        error,
        "",
        "Write an AgentPatch(...) expression with edits to fix this error.",
        "No markdown fencing.",
    ]

    match fn("\n".join(system_parts), "\n".join(user_parts)):
        case Err(e):
            logger.warning("Ouroboros model error: %s", e)
            return None
        case Ok(raw_text):
            pass

    try:
        patch = _parse_patch_response(raw_text)
    except ValueError as e:
        logger.warning("Ouroboros parse error: %s", e)
        return None

    logger.info(
        "Ouroboros patch: %d edit(s)",
        len(patch.edits),
    )

    match apply_patch(spec, patch):
        case Err(e):
            logger.warning("Ouroboros patch apply failed: %s", e)
            return None
        case Ok(corrected):
            return corrected


def _parse_patch_response(raw_text: str) -> AgentPatch:
    """Parse AgentPatch from model response."""
    import re

    edits: list[Edit] = []

    # Edit(old="...", new="...")
    for m in re.finditer(
        r"Edit\(\s*old\s*=\s*['\"](.*?)['\"]\s*,\s*new\s*=\s*['\"](.*?)['\"]",
        raw_text, re.DOTALL,
    ):
        edits.append(Edit(old=m.group(1), new=m.group(2)))

    if not edits:
        raise ValueError("No valid Edit entries found in response")

    return AgentPatch(edits=tuple(edits))


# ---------------------------------------------------------------------------
# Emit -- write agent to disk as .py
# ---------------------------------------------------------------------------


def emit_agent(
    spec: FourAgentSpec,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write a FourAgentSpec to disk as a .py file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in spec.title
    ).strip().replace(" ", "_").lower()

    if version > 0:
        filename = f"{safe_title}_v{version}.py"
    else:
        filename = f"{safe_title}.py"

    out_path = output_dir / filename
    out_path.write_text(spec.source)
    logger.info("Agent written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Load -- read .py back into FourAgentSpec
# ---------------------------------------------------------------------------


def load_agent(path: Path) -> Result[FourAgentSpec, str]:
    """Load a .py file into a FourAgentSpec."""
    if not path.exists():
        return Err(f"File not found: {path}")

    source = path.read_text()

    # Extract title from docstring or filename
    title = path.stem.replace("_", " ").title()

    # Try to extract variant from imports
    variant = "chat"
    if "litellm_toolcall_invoke" in source:
        variant = "toolcall"
    elif "http_response_invoke" in source:
        variant = "responses"

    return Ok(FourAgentSpec(
        title=title,
        description=f"Agent loaded from {path.name}",
        variant=variant,
        system_prompt="",
        source=source,
    ))
