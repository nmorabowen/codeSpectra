"""Exception hierarchy for codeSpectra.

The guiding rule: a design code that says "a site-specific study is required"
must never be answered with a silently-invented number. Those cases raise
:class:`SiteSpecificRequired`, which carries the clause that triggered it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .references import ClauseRef

__all__ = [
    "CodeSpectraError",
    "InvalidInput",
    "SiteSpecificRequired",
    "TableLookupError",
]


class CodeSpectraError(Exception):
    """Base class for every error raised by this library."""


class InvalidInput(CodeSpectraError, ValueError):
    """A user-supplied value is outside the domain the code permits."""


class TableLookupError(CodeSpectraError, KeyError):
    """A row/column requested from a code table does not exist."""

    def __str__(self) -> str:  # KeyError repr-quotes its arg; undo that.
        return self.args[0] if self.args else ""


class SiteSpecificRequired(CodeSpectraError):
    """The standard requires a site-specific study for this combination.

    Raised instead of returning a fabricated coefficient. Every construction
    site that can raise this also accepts an override so an engineer who *has*
    performed the study can proceed.

    Parameters
    ----------
    message
        Human-readable explanation of what triggered the requirement.
    refs
        Clauses that impose the requirement.
    remedy
        How the caller may legitimately proceed (a code-permitted exception,
        or the override keyword to pass).
    """

    def __init__(
        self,
        message: str,
        refs: Sequence[ClauseRef] = (),
        remedy: str | None = None,
    ) -> None:
        self.refs = tuple(refs)
        self.remedy = remedy
        parts = [message]
        if self.refs:
            parts.append("Required by: " + "; ".join(str(r) for r in self.refs))
        if remedy:
            parts.append(f"Remedy: {remedy}")
        super().__init__(" ".join(parts))
