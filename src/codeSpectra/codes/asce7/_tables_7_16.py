"""ASCE 7-16 site coefficient tables, transcribed from the standard.

Sources, verified against ASCE/SEI 7-16:
  * Table 11.4-1 Short-Period Site Coefficient, Fa   (p. 84)
  * Table 11.4-2 Long-Period Site Coefficient, Fv    (p. 84)
  * Table 11.9-1 Values of Vertical Coefficient Cv   (p. 87)

``None`` cells are the "See Section 11.4.8" entries, which defer to a ground
motion hazard analysis.
"""

from __future__ import annotations

from ...core.tables import InterpolatedTable
from ._shared import ref

EDITION = "7-16"

_SITE_SPECIFIC_REMEDY = (
    "Perform a ground motion hazard analysis per §21.2, or invoke a §11.4.8 "
    "exception via allow_site_specific_exception=True."
)

#: Table 11.4-1 — Short-period site coefficient Fa.
FA_TABLE = InterpolatedTable(
    name="Fa",
    row_label="site class",
    col_label="Ss",
    columns=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    rows={
        "A": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        "B": (0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        "C": (1.3, 1.3, 1.2, 1.2, 1.2, 1.2),
        "D": (1.6, 1.4, 1.2, 1.1, 1.0, 1.0),
        "E": (2.4, 1.7, 1.3, None, None, None),
        # Site Class F always defers to §11.4.8.
        "F": (None, None, None, None, None, None),
    },
    ref=ref(EDITION, "11.4.4", "Short-period site coefficient", table="11.4-1"),
    site_specific_remedy=_SITE_SPECIFIC_REMEDY,
)

#: Table 11.4-2 — Long-period site coefficient Fv.
FV_TABLE = InterpolatedTable(
    name="Fv",
    row_label="site class",
    col_label="S1",
    columns=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    rows={
        "A": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        "B": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        "C": (1.5, 1.5, 1.5, 1.5, 1.5, 1.4),
        # Footnote a: D cells at S1 >= 0.2 also trigger §11.4.8, but the
        # coefficient itself is tabulated, so the value stands and the
        # requirement is reported separately by ASCE7_16.site_specific_triggers.
        "D": (2.4, 2.2, 2.0, 1.9, 1.8, 1.7),
        "E": (4.2, None, None, None, None, None),
        "F": (None, None, None, None, None, None),
    },
    ref=ref(EDITION, "11.4.4", "Long-period site coefficient", table="11.4-2"),
    site_specific_remedy=_SITE_SPECIFIC_REMEDY,
)

#: Table 11.9-1 — Vertical coefficient Cv, grouped site classes.
CV_TABLE = InterpolatedTable(
    name="Cv",
    row_label="site class group",
    col_label="Ss",
    columns=(0.2, 0.3, 0.6, 1.0, 2.0),
    rows={
        "AB": (0.7, 0.8, 0.9, 0.9, 0.9),
        "C": (0.7, 0.8, 1.0, 1.1, 1.3),
        "DEF": (0.7, 0.9, 1.1, 1.3, 1.5),
    },
    ref=ref(EDITION, "11.9.2", "Vertical coefficient", table="11.9-1"),
)

#: Table 12.8-1 — Coefficient for upper limit on calculated period, Cu.
CU_TABLE = InterpolatedTable(
    name="Cu",
    row_label="applicability",
    col_label="SD1",
    columns=(0.1, 0.15, 0.2, 0.3, 0.4),
    rows={"all": (1.7, 1.6, 1.5, 1.4, 1.4)},
    ref=ref(EDITION, "12.8.2", "Upper limit coefficient on calculated period",
            table="12.8-1"),
)


def cv_row_for(site_class: str) -> str:
    """Map a site class to its Table 11.9-1 grouped row."""
    if site_class in ("A", "B"):
        return "AB"
    if site_class == "C":
        return "C"
    return "DEF"
