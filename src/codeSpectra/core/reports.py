"""Renderable calculation records.

A ``Report`` is the audit trail for a set of derived values: what went in,
what came out, and which clause authorised each step. It renders to plain
text or Markdown so it can be pasted into a calculation package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .references import ClauseRef

__all__ = ["Report", "ReportItem"]


def _format(value: float | str | bool | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.6g}"
    return str(value)


@dataclass(frozen=True, slots=True)
class ReportItem:
    """One line of a report: a named quantity, its value, and its authority."""

    symbol: str
    value: float | str | bool | None
    description: str = ""
    unit: str = ""
    ref: ClauseRef | None = None

    def __str__(self) -> str:
        value = _format(self.value)
        if self.unit:
            value = f"{value} {self.unit}"
        line = f"{self.symbol} = {value}"
        if self.description:
            line += f"  ({self.description})"
        if self.ref is not None:
            line += f"  [{self.ref}]"
        return line


@dataclass(frozen=True, slots=True)
class Report:
    """An ordered, immutable set of :class:`ReportItem` under a title."""

    title: str
    items: tuple[ReportItem, ...] = ()
    sections: tuple[Report, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __getitem__(self, symbol: str) -> float | str | bool | None:
        """Value of the item with this symbol, searching sub-sections too."""
        for item in self.items:
            if item.symbol == symbol:
                return item.value
        for section in self.sections:
            try:
                return section[symbol]
            except KeyError:
                continue
        raise KeyError(symbol)

    def values(self) -> dict[str, float | str | bool | None]:
        """Flatten every item, sub-sections included, into a dict."""
        out: dict[str, float | str | bool | None] = {}
        for section in self.sections:
            out.update(section.values())
        out.update({item.symbol: item.value for item in self.items})
        return out

    def refs(self) -> tuple[ClauseRef, ...]:
        """Every distinct citation used, in order of first appearance."""
        seen: dict[ClauseRef, None] = {}
        for section in self.sections:
            for ref in section.refs():
                seen.setdefault(ref, None)
        for item in self.items:
            if item.ref is not None:
                seen.setdefault(item.ref, None)
        return tuple(seen)

    # -- Rendering --------------------------------------------------------
    def to_text(self, indent: int = 0) -> str:
        """Plain-text rendering, suitable for a terminal or a code comment."""
        pad = " " * indent
        lines = [f"{pad}{self.title}", f"{pad}{'-' * len(self.title)}"]
        lines.extend(f"{pad}  {item}" for item in self.items)
        for section in self.sections:
            lines.append("")
            lines.append(section.to_text(indent + 2))
        if self.notes:
            lines.append("")
            lines.extend(f"{pad}  NOTE: {note}" for note in self.notes)
        return "\n".join(lines)

    def to_markdown(self, level: int = 2) -> str:
        """Markdown rendering with a value table per section."""
        lines = [f"{'#' * level} {self.title}", ""]
        if self.items:
            lines += ["| Symbol | Value | Description | Reference |",
                      "| --- | --- | --- | --- |"]
            for item in self.items:
                value = _format(item.value)
                if item.unit:
                    value = f"{value} {item.unit}"
                ref = str(item.ref) if item.ref is not None else ""
                lines.append(f"| `{item.symbol}` | {value} | {item.description} | {ref} |")
            lines.append("")
        for section in self.sections:
            lines.append(section.to_markdown(level + 1))
        for note in self.notes:
            lines.append(f"> **Note:** {note}")
            lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    # -- Construction helpers ---------------------------------------------
    def with_items(self, items: Sequence[ReportItem]) -> Report:
        """Return a copy with extra items appended."""
        return Report(self.title, self.items + tuple(items), self.sections, self.notes)

    def with_note(self, note: str) -> Report:
        """Return a copy with an extra note appended."""
        return Report(self.title, self.items, self.sections, (*self.notes, note))
