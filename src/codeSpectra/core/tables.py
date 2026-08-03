"""Straight-line-interpolated lookup tables, as printed in the standards.

Every site-coefficient table in ASCE 7 and NEC-SE-DS has the same shape: rows
keyed by site/soil class, columns keyed by a mapped ground-motion parameter,
with the instruction *"use straight-line interpolation for intermediate
values"* and some cells replaced by a pointer to a site-specific study.

That last part is the reason this class exists rather than a dict of floats.
A missing cell is represented by ``None`` and produces a
:class:`~codeSpectra.core.exceptions.SiteSpecificRequired` — including when it
merely *brackets* the requested value, because interpolating toward an
undefined cell is not a defined operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import SiteSpecificRequired, TableLookupError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .references import ClauseRef

__all__ = ["InterpolatedTable"]


@dataclass(frozen=True, slots=True)
class InterpolatedTable:
    """A row/column code table with linear interpolation across the columns.

    Parameters
    ----------
    name
        Coefficient produced, e.g. ``"Fa"``.
    row_label, col_label
        What the axes mean, used in error messages, e.g. ``"site class"`` and
        ``"Ss"``.
    columns
        Column breakpoints, ascending. Values below the first or above the
        last are clamped — the standards print these as ``<=`` and ``>=``.
    rows
        Mapping of row key to a tuple of cell values, one per column. ``None``
        marks a cell the standard defers to a site-specific study.
    ref
        Citation for the table.
    site_specific_remedy
        Text describing how a caller may legitimately proceed.
    """

    name: str
    row_label: str
    col_label: str
    columns: tuple[float, ...]
    rows: Mapping[str, tuple[float | None, ...]]
    ref: ClauseRef
    site_specific_remedy: str | None = None

    def __post_init__(self) -> None:
        cols = self.columns
        if len(cols) < 2:
            raise ValueError(f"Table {self.name} needs at least two columns.")
        if any(b <= a for a, b in pairwise(cols)):
            raise ValueError(f"Table {self.name} columns must be strictly ascending.")
        for key, values in self.rows.items():
            if len(values) != len(cols):
                raise ValueError(
                    f"Table {self.name} row {key!r} has {len(values)} cells, "
                    f"expected {len(cols)}."
                )

    @property
    def row_keys(self) -> tuple[str, ...]:
        """Row keys defined in the table, in insertion order."""
        return tuple(self.rows)

    def lookup(self, row: str, value: float) -> float:
        """Interpolate the coefficient for ``row`` at ``value``.

        Raises
        ------
        TableLookupError
            ``row`` is not a row of this table.
        SiteSpecificRequired
            The cell, or a cell bracketing ``value``, is deferred by the
            standard to a site-specific study.
        """
        try:
            cells = self.rows[row]
        except KeyError:
            raise TableLookupError(
                f"{self.row_label} {row!r} is not defined in table "
                f"{self.name} ({self.ref}). Defined: {list(self.rows)}."
            ) from None

        x = float(value)
        cols = self.columns
        # Clamp: the printed tables bound the first/last column with <= and >=.
        if x <= cols[0]:
            return self._cell(row, cells, 0, x)
        if x >= cols[-1]:
            return self._cell(row, cells, len(cols) - 1, x)

        hi = int(np.searchsorted(np.asarray(cols), x, side="left"))
        lo = hi - 1
        if x == cols[lo]:
            return self._cell(row, cells, lo, x)
        y_lo = self._cell(row, cells, lo, x)
        y_hi = self._cell(row, cells, hi, x)
        span = cols[hi] - cols[lo]
        return y_lo + (y_hi - y_lo) * (x - cols[lo]) / span

    def _cell(
        self,
        row: str,
        cells: Sequence[float | None],
        index: int,
        requested: float,
    ) -> float:
        cell = cells[index]
        if cell is None:
            raise SiteSpecificRequired(
                f"{self.name} is not tabulated for {self.row_label} {row!r} at "
                f"{self.col_label} = {requested:g} "
                f"(table cell at {self.col_label} = {self.columns[index]:g}).",
                refs=(self.ref,),
                remedy=self.site_specific_remedy,
            )
        return float(cell)

    def is_defined(self, row: str, value: float) -> bool:
        """Whether :meth:`lookup` would succeed for this combination."""
        try:
            self.lookup(row, value)
        except (SiteSpecificRequired, TableLookupError):
            return False
        return True
