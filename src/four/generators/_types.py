"""
Shared foundation types for all generators.

FP discipline: this module imports only stdlib. No external dependencies.
All data is frozen/immutable. Validation functions return Result[T, E].
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Generic, Optional, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")


# ============================================================================
# Result type
# ============================================================================


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


@dataclass(frozen=True)
class Cycle:
    """Partial success: non-deterministic step completed, re-plan needed."""
    facts: dict = field(default_factory=dict)
    message: str = ""


Result = Union[Ok[T], Err[E]]


# ============================================================================
# Domain context
# ============================================================================


@dataclass(frozen=True)
class DomainSection:
    """A named section of domain knowledge for the model's system prompt."""
    heading: str
    content: str


@dataclass(frozen=True)
class FileSpec:
    """A file declared by the model as part of the solution."""
    path: str
    content: str
    description: str


# ============================================================================
# Generation context
# ============================================================================


@dataclass(frozen=True)
class GenerationContext:
    """Immutable accumulator -- everything the model knows."""
    domain_context: tuple[DomainSection, ...] = ()
    available_packages: str = ""
    feedback: tuple[str, ...] = ()
    user_prompt: Optional[str] = None
    default_task: Optional[str] = None

    def with_domain(self, section: DomainSection) -> GenerationContext:
        return replace(self, domain_context=(*self.domain_context, section))

    def with_feedback(self, msg: str) -> GenerationContext:
        return replace(self, feedback=(*self.feedback, msg))

    def with_prompt(self, prompt: str) -> GenerationContext:
        return replace(self, user_prompt=prompt, feedback=())


# ============================================================================
# AskFn -- the injected model invocation contract
# ============================================================================

AskFn = Callable[[str, str], Result]


# ============================================================================
# Generation report
# ============================================================================


@dataclass(frozen=True)
class GenerationReport:
    """Provenance record for a generated artifact."""
    version: int = 0
    rounds: int = 0
    ouroboros_fixes: int = 0
    outcome: str = "success"
    claim: Optional[str] = None
    validators: tuple[str, ...] = ()
    user_prompt: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {
            "version": self.version,
            "rounds": self.rounds,
            "ouroboros_fixes": self.ouroboros_fixes,
            "outcome": self.outcome,
            "validators": list(self.validators),
        }
        if self.claim:
            d["claim"] = self.claim
        if self.user_prompt:
            d["user_prompt"] = self.user_prompt
        return d
