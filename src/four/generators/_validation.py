"""
Reusable Python source validation.

Pure functions operating on raw strings via ast module. No notebook types,
no spec types -- just Python source code analysis.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass

from ._types import Err, Ok, Result

_BUILTINS = frozenset(vars(builtins))

_CELL_INDEX_RE = re.compile(r"cells\[(\d+)\]")


# ============================================================================
# AST definition collector
# ============================================================================


def _names_from_target(node: ast.AST) -> list[str]:
    """Extract assigned names from an assignment target (handles unpacking)."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in node.elts:
            names.extend(_names_from_target(elt))
        return names
    if isinstance(node, ast.Starred):
        return _names_from_target(node.value)
    return []


class DefCollector(ast.NodeVisitor):
    """Collects variable definitions with conditionality tracking."""

    def __init__(self) -> None:
        self.defs: dict[str, bool] = {}
        self._depth = 0

    def _record(self, name: str, conditional: bool) -> None:
        existing = self.defs.get(name)
        if existing is not None and not existing:
            return
        self.defs[name] = conditional

    def _record_targets(self, targets: list[ast.AST]) -> None:
        for t in targets:
            for name in _names_from_target(t):
                self._record(name, conditional=self._depth > 0)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target:
            for name in _names_from_target(node.target):
                self._record(name, conditional=self._depth > 0)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _names_from_target(node.target):
            self._record(name, conditional=self._depth > 0)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node.name, conditional=self._depth > 0)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node.name, conditional=self._depth > 0)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node.name, conditional=self._depth > 0)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.asname or alias.name.split(".")[0],
                         conditional=self._depth > 0)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._record(alias.asname or alias.name,
                         conditional=self._depth > 0)

    def visit_For(self, node: ast.For) -> None:
        for name in _names_from_target(node.target):
            self._record(name, conditional=False)
        self._depth += 1
        for child in node.body + node.orelse:
            self.visit(child)
        self._depth -= 1

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars:
                for name in _names_from_target(item.optional_vars):
                    self._record(name, conditional=self._depth > 0)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1


def collect_definitions(tree: ast.Module) -> dict[str, bool]:
    """Return {name: conditional} for all names defined in a source."""
    collector = DefCollector()
    collector.visit(tree)
    return collector.defs


def collect_references(tree: ast.Module) -> set[str]:
    """Return all names used in Load context."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


# ============================================================================
# Cross-source Python validation
# ============================================================================


@dataclass(frozen=True)
class _VarDef:
    source_index: int
    conditional: bool


def validate_python_sources(
    sources: tuple[str, ...],
    *,
    label: str = "cells",
) -> Result[None, str]:
    """Cross-source Python validation: syntax, variable flow."""
    errors: list[str] = []
    env: dict[str, _VarDef] = {}

    for i, source in enumerate(sources):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            loc = f" at line {exc.lineno}" if exc.lineno else ""
            errors.append(f"{label}[{i}]: SyntaxError{loc}: {exc.msg}")
            continue

        cell_defs = collect_definitions(tree)
        cell_refs = collect_references(tree)

        for name in cell_refs:
            if name in _BUILTINS or name in cell_defs:
                continue
            prior = env.get(name)
            if prior is not None and prior.conditional:
                errors.append(
                    f"{label}[{i}]: '{name}' may be undefined -- "
                    f"defined conditionally in {label}[{prior.source_index}]"
                )

        for name, conditional in cell_defs.items():
            existing = env.get(name)
            if existing and not existing.conditional and conditional:
                continue
            env[name] = _VarDef(source_index=i, conditional=conditional)

    if errors:
        return Err("; ".join(errors))
    return Ok(None)


def summarize_variable_flow(
    sources: tuple[str, ...],
    *,
    label: str = "cells",
) -> str:
    """Produce a summary of cross-source variable flow."""
    env: dict[str, _VarDef] = {}
    lines: list[str] = [f"Cross-{label} variable flow:"]

    for i, source in enumerate(sources):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            lines.append(f"  {label}[{i}]: <syntax error, skipped>")
            continue

        cell_defs = collect_definitions(tree)
        cell_refs = collect_references(tree)

        unconditional = sorted(n for n, cond in cell_defs.items() if not cond)
        conditional = sorted(n for n, cond in cell_defs.items() if cond)

        cross_refs: list[str] = []
        for name in sorted(cell_refs):
            if name in _BUILTINS or name in cell_defs:
                continue
            prior = env.get(name)
            if prior is not None:
                qualifier = "conditional" if prior.conditional else "ok"
                cross_refs.append(
                    f"{name} (from {label}[{prior.source_index}], {qualifier})"
                )

        if unconditional or conditional or cross_refs:
            lines.append(f"  {label}[{i}]:")
            if unconditional:
                lines.append(f"    defines: {', '.join(unconditional)}")
            if conditional:
                lines.append(f"    defines conditionally: {', '.join(conditional)}")
            if cross_refs:
                lines.append(f"    references: {', '.join(cross_refs)}")

        for name, cond in cell_defs.items():
            existing = env.get(name)
            if existing and not existing.conditional and cond:
                continue
            env[name] = _VarDef(source_index=i, conditional=cond)

    return "\n".join(lines)


def extract_first_cell_index(error: str) -> int | None:
    """Extract the first cells[N] index from an error string."""
    m = _CELL_INDEX_RE.search(error)
    return int(m.group(1)) if m else None
