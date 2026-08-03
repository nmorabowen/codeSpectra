"""Machinery common to every ASCE 7 edition.

The two-period spectrum shape (Fig. 11.4-1) is unchanged from 7-05 through
7-22; what changes between editions is how ``SDS`` and ``SD1`` are obtained.
Keeping the shape here means the edition modules only supply parameters.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ...core.control import ControlPeriods
from ...core.exceptions import InvalidInput
from ...core.references import ClauseRef
from ...core.spectrum import AnalyticSpectrum, SpectrumKind, SpectrumMeta

__all__ = [
    "RiskCategory",
    "SeismicDesignCategory",
    "SiteClass",
    "control_periods",
    "ref",
    "seismic_design_category",
    "two_period_evaluator",
    "two_period_spectrum",
]

STANDARD = "ASCE/SEI 7"


class SiteClass(str, Enum):
    """Site class per ASCE 7 Chapter 20.

    Classes ``BC``, ``CD`` and ``DE`` were introduced in ASCE 7-22 (Table
    20.2-1); editions 7-10 and 7-16 accept only A-F. Each edition class
    validates which members it permits.
    """

    A = "A"
    """Hard rock."""
    B = "B"
    """Medium hard rock."""
    BC = "BC"
    """Soft rock (7-22 and later)."""
    C = "C"
    """Very dense sand or hard clay."""
    CD = "CD"
    """Dense sand or very stiff clay (7-22 and later)."""
    D = "D"
    """Stiff soil."""
    DE = "DE"
    """Medium stiff soil (7-22 and later)."""
    E = "E"
    """Soft clay soil."""
    F = "F"
    """Soils requiring site response analysis."""


def ref(
    edition: str,
    clause: str,
    description: str = "",
    *,
    equation: str | None = None,
    table: str | None = None,
    figure: str | None = None,
) -> ClauseRef:
    """Build an ASCE 7 citation for ``edition`` (e.g. ``"7-16"``)."""
    return ClauseRef(
        standard=STANDARD,
        edition=edition,
        clause=clause,
        equation=equation,
        table=table,
        figure=figure,
        description=description,
    )


class RiskCategory(str, Enum):
    """Risk category per ASCE 7 Table 1.5-1."""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"

    @property
    def importance_factor(self) -> float:
        """Seismic importance factor ``Ie``, Table 1.5-2."""
        return {"I": 1.00, "II": 1.00, "III": 1.25, "IV": 1.50}[self.value]


class SeismicDesignCategory(str, Enum):
    """Seismic design category per §11.6."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"

    @property
    def severity(self) -> int:
        """0 for A through 5 for F; higher is more severe."""
        return "ABCDEF".index(self.value)


def control_periods(SDS: float, SD1: float, TL: float) -> ControlPeriods:
    """``T0 = 0.2 SD1/SDS``, ``Ts = SD1/SDS``, and ``TL`` (§11.4.6)."""
    if SDS <= 0.0:
        raise InvalidInput(f"SDS must be positive, got {SDS}.")
    if SD1 < 0.0:
        raise InvalidInput(f"SD1 must be non-negative, got {SD1}.")
    if TL <= 0.0:
        raise InvalidInput(f"TL must be positive, got {TL}.")
    Ts = SD1 / SDS
    return ControlPeriods(T0=0.2 * Ts, Ts=Ts, TL=float(TL))


def two_period_evaluator(
    SDS: float, SD1: float, TL: float
) -> tuple[ControlPeriods, _TwoPeriod]:
    """Return the control periods and a callable implementing Fig. 11.4-1."""
    cp = control_periods(SDS, SD1, TL)
    return cp, _TwoPeriod(SDS, SD1, float(TL), cp.T0, cp.Ts)


class _TwoPeriod:
    """Fig. 11.4-1, Eqs. 11.4-5 through 11.4-7 (7-16 numbering).

    Ascending branch below ``T0``, plateau to ``Ts``, ``1/T`` decay to ``TL``,
    then ``1/T^2``. Vectorised and branch-exact at the corners.
    """

    __slots__ = ("SD1", "SDS", "T0", "TL", "Ts")

    def __init__(self, SDS: float, SD1: float, TL: float, T0: float, Ts: float) -> None:
        self.SDS = SDS
        self.SD1 = SD1
        self.TL = TL
        self.T0 = T0
        self.Ts = Ts

    def __call__(self, T: NDArray[np.float64]) -> NDArray[np.float64]:
        T = np.asarray(T, dtype=float)
        out = np.full(T.shape, self.SDS, dtype=float)

        if self.T0 > 0.0:
            ascending = T < self.T0
            out[ascending] = self.SDS * (0.4 + 0.6 * T[ascending] / self.T0)

        # Guard against T == 0 in the decay branches.
        safe = np.where(T > 0.0, T, 1.0)
        mid = (self.Ts < T) & (T <= self.TL)
        out[mid] = self.SD1 / safe[mid]
        long = T > self.TL
        out[long] = self.SD1 * self.TL / safe[long] ** 2
        return out


def two_period_spectrum(
    *,
    edition: str,
    SDS: float,
    SD1: float,
    TL: float,
    kind: SpectrumKind = SpectrumKind.DESIGN,
    label: str = "",
    parameters: dict[str, float | str] | None = None,
    refs: tuple[ClauseRef, ...] = (),
    t_max: float | None = None,
) -> AnalyticSpectrum:
    """Build the two-period design response spectrum of Fig. 11.4-1."""
    cp, evaluator = two_period_evaluator(SDS, SD1, TL)
    params: dict[str, float | str] = {"SDS": SDS, "SD1": SD1, "TL": TL}
    if parameters:
        params.update(parameters)
    meta = SpectrumMeta(
        standard=STANDARD,
        edition=edition,
        kind=kind,
        damping=0.05,
        label=label or f"{STANDARD} {edition} {kind.value} spectrum",
        control_periods=cp,
        parameters=params,
        refs=refs,
    )
    upper = t_max if t_max is not None else max(6.0, 1.25 * TL)
    return AnalyticSpectrum(evaluator, meta=meta, t_max=upper)


# -- Seismic design category (§11.6) --------------------------------------

_SDS_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.167, "A", "A"),
    (0.33, "B", "C"),
    (0.50, "C", "D"),
    (float("inf"), "D", "D"),
)

_SD1_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.067, "A", "A"),
    (0.133, "B", "C"),
    (0.20, "C", "D"),
    (float("inf"), "D", "D"),
)


def seismic_design_category(
    SDS: float,
    SD1: float,
    S1: float,
    risk_category: RiskCategory,
) -> SeismicDesignCategory:
    """Seismic design category per §11.6, Tables 11.6-1 and 11.6-2.

    Applies the ``S1 >= 0.75`` override first (SDC E for Risk Category I-III,
    F for IV), then takes the more severe of the two table results.
    """
    if S1 >= 0.75:
        return (
            SeismicDesignCategory.F
            if risk_category is RiskCategory.IV
            else SeismicDesignCategory.E
        )
    idx = 1 if risk_category is RiskCategory.IV else 0
    from_sds = next(row[1 + idx] for row in _SDS_BANDS if row[0] > SDS)
    from_sd1 = next(row[1 + idx] for row in _SD1_BANDS if row[0] > SD1)
    return max(
        (SeismicDesignCategory(from_sds), SeismicDesignCategory(from_sd1)),
        key=lambda sdc: sdc.severity,
    )
