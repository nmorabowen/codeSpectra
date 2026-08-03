"""Traceable citations attached to every derived quantity.

A design number without its clause is an opinion. Every coefficient, control
period, and spectrum this library produces carries a :class:`ClauseRef` so a
calculation package can be rendered straight from the objects.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ClauseRef"]


@dataclass(frozen=True, slots=True)
class ClauseRef:
    """A pointer into a published standard.

    Parameters
    ----------
    standard
        Short name of the document, e.g. ``"ASCE/SEI 7"`` or ``"NEC-SE-DS"``.
    edition
        Edition or year, e.g. ``"7-16"`` or ``"2015"``.
    clause
        Section number, e.g. ``"11.4.4"`` or ``"3.3.1"``.
    equation, table, figure
        Optional specific artifact within the clause.
    description
        What the citation establishes, in one phrase.
    """

    standard: str
    edition: str
    clause: str = ""
    equation: str | None = None
    table: str | None = None
    figure: str | None = None
    description: str = ""

    def __str__(self) -> str:
        head = f"{self.standard} {self.edition}"
        bits: list[str] = []
        if self.clause:
            bits.append(f"§{self.clause}")
        if self.equation:
            bits.append(f"Eq. {self.equation}")
        if self.table:
            bits.append(f"Table {self.table}")
        if self.figure:
            bits.append(f"Fig. {self.figure}")
        cite = f"{head} {', '.join(bits)}" if bits else head
        return f"{cite} — {self.description}" if self.description else cite

    def with_description(self, description: str) -> ClauseRef:
        """Return a copy carrying a different description."""
        return ClauseRef(
            standard=self.standard,
            edition=self.edition,
            clause=self.clause,
            equation=self.equation,
            table=self.table,
            figure=self.figure,
            description=description,
        )
