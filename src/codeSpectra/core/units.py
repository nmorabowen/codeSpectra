"""Unit handling.

Spectral acceleration is stored internally **in g, always**. Every code this
library implements defines ``Sa`` as a fraction of gravity, so g is the one
convention that needs no translation at the point of definition. Conversion
happens only at the export/plot boundary.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

__all__ = ["STANDARD_GRAVITY", "AccelUnit", "from_g"]

#: Standard gravity, m/s^2 (CGPM 1901; the value ASCE/NEC design practice uses).
STANDARD_GRAVITY = 9.80665


class AccelUnit(str, Enum):
    """Acceleration units available at the export boundary."""

    G = "g"
    M_S2 = "m/s2"
    CM_S2 = "cm/s2"
    IN_S2 = "in/s2"
    FT_S2 = "ft/s2"

    @property
    def factor_from_g(self) -> float:
        """Multiplier converting a value in g to this unit."""
        return _FACTORS[self]

    @property
    def label(self) -> str:
        """Axis label form, e.g. ``"m/s²"``."""
        return _LABELS[self]


_FACTORS: dict[AccelUnit, float] = {
    AccelUnit.G: 1.0,
    AccelUnit.M_S2: STANDARD_GRAVITY,
    AccelUnit.CM_S2: STANDARD_GRAVITY * 100.0,
    AccelUnit.IN_S2: STANDARD_GRAVITY / 0.0254,
    AccelUnit.FT_S2: STANDARD_GRAVITY / 0.3048,
}

_LABELS: dict[AccelUnit, str] = {
    AccelUnit.G: "g",
    AccelUnit.M_S2: "m/s²",
    AccelUnit.CM_S2: "cm/s²",
    AccelUnit.IN_S2: "in/s²",
    AccelUnit.FT_S2: "ft/s²",
}


def from_g(values: NDArray[np.float64], unit: AccelUnit | str) -> NDArray[np.float64]:
    """Convert an array of accelerations expressed in g to ``unit``."""
    return values * AccelUnit(unit).factor_from_g
