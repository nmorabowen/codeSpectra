"""ASCE 7 equivalent lateral force procedure (§12.8).

Covers the approximate period ``Ta`` (§12.8.2.1), the seismic response
coefficient ``Cs`` with all four limits (§12.8.1.1), base shear ``V``
(§12.8.1) and the vertical force distribution (§12.8.3).

Clause numbering below is 7-16; 7-10 and 7-22 use the same §12.8 numbering for
these equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.exceptions import InvalidInput
from ...core.reports import Report, ReportItem
from ._shared import ref

if TYPE_CHECKING:
    from ...core.references import ClauseRef

__all__ = [
    "CU_TABLE",
    "ELFResult",
    "StructureType",
    "approximate_period",
    "base_shear",
    "seismic_response_coefficient",
    "upper_limit_coefficient",
    "vertical_distribution",
    "vertical_distribution_exponent",
]

EDITION = "7-16"


class StructureType(str, Enum):
    """Structural system for Table 12.8-2 approximate period parameters."""

    STEEL_MOMENT_FRAME = "steel_moment_frame"
    CONCRETE_MOMENT_FRAME = "concrete_moment_frame"
    STEEL_EBF = "steel_eccentrically_braced_frame"
    STEEL_BRBF = "steel_buckling_restrained_braced_frame"
    OTHER = "other"

    @property
    def parameters(self) -> tuple[float, float, float]:
        """``(Ct_imperial, Ct_metric, x)`` from Table 12.8-2.

        ``Ct_imperial`` applies with ``hn`` in feet, ``Ct_metric`` with ``hn``
        in metres.
        """
        return _TABLE_12_8_2[self.value]


#: Table 12.8-2 — Ct (ft), Ct (m), and exponent x.
_TABLE_12_8_2: dict[str, tuple[float, float, float]] = {
    "steel_moment_frame": (0.028, 0.0724, 0.8),
    "concrete_moment_frame": (0.016, 0.0466, 0.9),
    "steel_eccentrically_braced_frame": (0.03, 0.0731, 0.75),
    "steel_buckling_restrained_braced_frame": (0.03, 0.0731, 0.75),
    "other": (0.02, 0.0488, 0.75),
}

#: Table 12.8-1 — upper limit coefficient Cu, keyed by SD1 breakpoints.
CU_TABLE: tuple[tuple[float, float], ...] = (
    (0.1, 1.7),
    (0.15, 1.6),
    (0.2, 1.5),
    (0.3, 1.4),
    (0.4, 1.4),
)


def upper_limit_coefficient(SD1: float) -> float:
    """Coefficient ``Cu`` for the upper limit on the calculated period.

    Table 12.8-1, with straight-line interpolation and clamping at the ends
    (the table is printed with ``<= 0.1`` and ``>= 0.4`` bounds).
    """
    xs = np.array([row[0] for row in CU_TABLE])
    ys = np.array([row[1] for row in CU_TABLE])
    return float(np.interp(SD1, xs, ys))


def approximate_period(
    hn: float,
    structure_type: StructureType | str = StructureType.OTHER,
    *,
    metric: bool = True,
) -> float:
    """Approximate fundamental period ``Ta = Ct hn**x`` (Eq. 12.8-7).

    Parameters
    ----------
    hn
        Structural height above the base, in metres if ``metric`` else feet.
    structure_type
        Row of Table 12.8-2.
    metric
        True if ``hn`` is in metres. Selects the parenthesised metric ``Ct``.
    """
    if hn <= 0.0:
        raise InvalidInput("hn must be positive.")
    Ct_ft, Ct_m, x = StructureType(structure_type).parameters
    Ct = Ct_m if metric else Ct_ft
    return float(Ct * hn**x)


def seismic_response_coefficient(
    *,
    SDS: float,
    SD1: float,
    S1: float,
    T: float,
    TL: float,
    R: float,
    Ie: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Seismic response coefficient ``Cs`` (§12.8.1.1), with all limits.

    Returns ``(Cs, breakdown)`` where ``breakdown`` reports each governing
    equation's value so a report can show which one controlled.

    Notes
    -----
    ``Cs = SDS/(R/Ie)`` (Eq. 12.8-2), capped by Eq. 12.8-3 for ``T <= TL`` or
    Eq. 12.8-4 for ``T > TL``, floored by Eq. 12.8-5 (``0.044 SDS Ie >= 0.01``)
    and, where ``S1 >= 0.6``, additionally floored by Eq. 12.8-6.
    """
    for name, value in (("R", R), ("Ie", Ie), ("T", T), ("TL", TL)):
        if value <= 0.0:
            raise InvalidInput(f"{name} must be positive, got {value}.")
    ratio = R / Ie

    cs_base = SDS / ratio
    cs_cap = (SD1 / (T * ratio)) if T <= TL else (SD1 * TL / (T**2 * ratio))
    cs_min = max(0.044 * SDS * Ie, 0.01)
    cs_min_s1 = 0.5 * S1 / ratio if S1 >= 0.6 else 0.0

    Cs = min(cs_base, cs_cap)
    Cs = max(Cs, cs_min, cs_min_s1)

    breakdown = {
        "Cs_eq_12.8-2": cs_base,
        "Cs_cap_eq_12.8-3/4": cs_cap,
        "Cs_min_eq_12.8-5": cs_min,
        "Cs_min_eq_12.8-6": cs_min_s1,
        "Cs": Cs,
    }
    return Cs, breakdown


def vertical_distribution_exponent(T: float) -> float:
    """Distribution exponent ``k`` (§12.8.3).

    1.0 for ``T <= 0.5`` s, 2.0 for ``T >= 2.5`` s, linearly interpolated
    between.
    """
    if T <= 0.5:
        return 1.0
    if T >= 2.5:
        return 2.0
    return 1.0 + (T - 0.5) / 2.0


def vertical_distribution(
    V: float,
    weights: ArrayLike,
    heights: ArrayLike,
    T: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distribute ``V`` over the storeys (Eqs. 12.8-11 and 12.8-12).

    Parameters
    ----------
    V
        Base shear.
    weights, heights
        Storey seismic weights ``wx`` and heights above base ``hx``, in
        matching order and consistent units.
    T
        Fundamental period, used for the exponent ``k``.

    Returns
    -------
    Fx, Vx
        Storey forces and the storey shear at each level (cumulative from the
        top down), in the same order as the inputs.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=float))
    h = np.atleast_1d(np.asarray(heights, dtype=float))
    if w.shape != h.shape:
        raise InvalidInput(
            f"weights and heights length mismatch: {w.size} vs {h.size}."
        )
    if np.any(w < 0.0):
        raise InvalidInput("Storey weights must be non-negative.")
    if np.any(h < 0.0):
        raise InvalidInput("Storey heights must be non-negative.")

    k = vertical_distribution_exponent(T)
    numerator = w * h**k
    total = float(np.sum(numerator))
    if total <= 0.0:
        raise InvalidInput(
            "Sum of w*h^k is zero; check the storey weights and heights."
        )
    Cvx = numerator / total
    Fx = Cvx * V

    # Storey shear accumulates from the roof downward.
    order = np.argsort(h)
    Vx = np.empty_like(Fx)
    running = 0.0
    for idx in order[::-1]:
        running += Fx[idx]
        Vx[idx] = running
    return Fx, Vx


@dataclass(frozen=True, slots=True)
class ELFResult:
    """Outcome of an ASCE 7 §12.8 equivalent lateral force calculation."""

    V: float
    Cs: float
    T: float
    Ta: float
    k: float
    breakdown: dict[str, float]
    Fx: NDArray[np.float64] | None = None
    Vx: NDArray[np.float64] | None = None

    @property
    def governing_equation(self) -> str:
        """Which §12.8.1.1 equation set ``Cs``."""
        eps = 1e-12
        for name in ("Cs_min_eq_12.8-6", "Cs_min_eq_12.8-5",
                     "Cs_cap_eq_12.8-3/4", "Cs_eq_12.8-2"):
            if abs(self.breakdown[name] - self.Cs) < eps:
                return name
        return "Cs_eq_12.8-2"

    def report(self, *, edition: str = EDITION) -> Report:
        """A citation-carrying record of the ELF calculation."""

        def r(clause: str, description: str, **kw: str) -> ClauseRef:
            return ref(edition, clause, description, **kw)

        items = [
            ReportItem("Ta", self.Ta, "Approximate fundamental period", "s",
                       r("12.8.2.1", "Approximate fundamental period",
                         equation="12.8-7")),
            ReportItem("T", self.T, "Fundamental period used", "s",
                       r("12.8.2", "Period determination")),
            ReportItem("Cs", self.Cs, f"Governed by {self.governing_equation}", "",
                       r("12.8.1.1", "Seismic response coefficient",
                         equation="12.8-2")),
            ReportItem("V", self.V, "Seismic base shear, Cs x W", "",
                       r("12.8.1", "Seismic base shear", equation="12.8-1")),
            ReportItem("k", self.k, "Vertical distribution exponent", "",
                       r("12.8.3", "Vertical distribution of seismic forces")),
        ]
        extra = [
            ReportItem(name, value, "", "",
                       r("12.8.1.1", "Seismic response coefficient limits"))
            for name, value in self.breakdown.items()
            if name != "Cs"
        ]
        return Report(
            title=f"ASCE/SEI {edition} equivalent lateral force (§12.8)",
            items=tuple(items),
            sections=(Report("Cs limit checks", tuple(extra)),),
        )

    def __str__(self) -> str:
        return self.report().to_text()


def base_shear(
    *,
    W: float,
    SDS: float,
    SD1: float,
    S1: float,
    TL: float,
    R: float,
    Ie: float = 1.0,
    T: float | None = None,
    hn: float | None = None,
    structure_type: StructureType | str = StructureType.OTHER,
    metric: bool = True,
    apply_upper_limit: bool = True,
    weights: ArrayLike | None = None,
    heights: ArrayLike | None = None,
) -> ELFResult:
    """Seismic base shear ``V = Cs W`` per §12.8.1.

    Parameters
    ----------
    W
        Effective seismic weight (§12.7.2).
    T
        Fundamental period from analysis. If omitted, ``Ta`` is used directly,
        which §12.8.2 permits. If supplied together with ``hn`` and
        ``apply_upper_limit``, it is capped at ``Cu * Ta`` per §12.8.2.
    hn
        Structural height, needed to compute ``Ta``.
    weights, heights
        Supply both to also return the vertical force distribution.
    """
    if W <= 0.0:
        raise InvalidInput("W must be positive.")
    if T is None and hn is None:
        raise InvalidInput("Supply the fundamental period T, the height hn, or both.")

    Ta = approximate_period(hn, structure_type, metric=metric) if hn is not None else T
    assert Ta is not None

    if T is None:
        T_used = Ta
    elif apply_upper_limit and hn is not None:
        T_used = min(T, upper_limit_coefficient(SD1) * Ta)
    else:
        T_used = T

    Cs, breakdown = seismic_response_coefficient(
        SDS=SDS, SD1=SD1, S1=S1, T=T_used, TL=TL, R=R, Ie=Ie
    )
    V = Cs * W
    k = vertical_distribution_exponent(T_used)

    Fx = Vx = None
    if weights is not None and heights is not None:
        Fx, Vx = vertical_distribution(V, weights, heights, T_used)

    return ELFResult(V=V, Cs=Cs, T=T_used, Ta=Ta, k=k,
                     breakdown=breakdown, Fx=Fx, Vx=Vx)
