"""NEC-SE-DS 2015 diseño basado en fuerzas (DBF), §6.3.

Base shear (§6.3.2)::

    V = I * Sa(Ta) * W / (R * phi_P * phi_E)

Period (§6.3.3): Método 1 gives ``Ta = Ct * hn**alpha``; Método 2 (Rayleigh)
may refine it but must not exceed 1.30 times the Método 1 value.

Vertical distribution (§6.3.5) uses ``k`` from the period-dependent table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.exceptions import InvalidInput
from ...core.references import ClauseRef
from ...core.reports import Report, ReportItem
from ...core.spectrum import Spectrum
from . import _tables as NT

__all__ = [
    "NECELFResult",
    "approximate_period",
    "base_shear",
    "limit_period_method_2",
    "vertical_distribution",
    "vertical_distribution_exponent",
]

#: Método 2 must not exceed Método 1 by more than 30% (§6.3.3b).
METHOD_2_MAX_RATIO = 1.30


def _ref(clause: str, description: str, **kw: str) -> ClauseRef:
    return ClauseRef(standard=NT.STANDARD, edition=NT.EDITION, clause=clause,
                     description=description, **kw)


def approximate_period(hn: float, structure_type: str) -> float:
    """Método 1 approximate period ``Ta = Ct hn**alpha`` (§6.3.3a).

    Parameters
    ----------
    hn
        Building height above the base, in metres.
    structure_type
        One of the keys of
        :data:`codeSpectra.codes.nec._tables.PERIOD_COEFFICIENTS`:
        ``acero_sin_arriostramientos``, ``acero_con_arriostramientos``,
        ``hormigon_porticos``, ``hormigon_con_muros``.
    """
    if hn <= 0.0:
        raise InvalidInput("hn must be positive (metres).")
    try:
        Ct, alpha, _ = NT.PERIOD_COEFFICIENTS[structure_type]
    except KeyError:
        raise InvalidInput(
            f"Unknown structure_type {structure_type!r}. Valid options: "
            f"{list(NT.PERIOD_COEFFICIENTS)}."
        ) from None
    return float(Ct * hn**alpha)


def limit_period_method_2(T_method_2: float, Ta_method_1: float) -> float:
    """Cap a Método 2 period at 1.30x the Método 1 value (§6.3.3b)."""
    if T_method_2 <= 0.0 or Ta_method_1 <= 0.0:
        raise InvalidInput("Periods must be positive.")
    return min(T_method_2, METHOD_2_MAX_RATIO * Ta_method_1)


def vertical_distribution_exponent(T: float) -> float:
    """Distribution exponent ``k`` (§6.3.5).

    1.0 for ``T <= 0.5`` s, ``0.75 + 0.50 T`` for ``0.5 < T <= 2.5`` s, and
    2.0 above that.
    """
    if T <= 0.5:
        return 1.0
    if T > 2.5:
        return 2.0
    return 0.75 + 0.50 * T


def vertical_distribution(
    V: float,
    weights: ArrayLike,
    heights: ArrayLike,
    T: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distribute ``V`` over the storeys per §6.3.5.

    Returns ``(Fx, Vx)`` — storey forces and storey shears, input order.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=float))
    h = np.atleast_1d(np.asarray(heights, dtype=float))
    if w.shape != h.shape:
        raise InvalidInput(
            f"weights and heights length mismatch: {w.size} vs {h.size}."
        )
    if np.any(w < 0.0) or np.any(h < 0.0):
        raise InvalidInput("Storey weights and heights must be non-negative.")

    k = vertical_distribution_exponent(T)
    numerator = w * h**k
    total = float(np.sum(numerator))
    if total <= 0.0:
        raise InvalidInput("Sum of w*h^k is zero; check weights and heights.")
    Fx = V * numerator / total

    order = np.argsort(h)
    Vx = np.empty_like(Fx)
    running = 0.0
    for idx in order[::-1]:
        running += Fx[idx]
        Vx[idx] = running
    return Fx, Vx


@dataclass(frozen=True, slots=True)
class NECELFResult:
    """Outcome of a NEC-SE-DS §6.3 base shear calculation."""

    V: float
    Sa_Ta: float
    Ta: float
    k: float
    R: float
    phi_p: float
    phi_e: float
    I: float
    W: float
    Fx: NDArray[np.float64] | None = None
    Vx: NDArray[np.float64] | None = None

    @property
    def shear_coefficient(self) -> float:
        """``V / W``, the fraction of seismic weight applied laterally."""
        return self.V / self.W

    def report(self) -> Report:
        """A citation-carrying record of the base shear calculation."""
        items = [
            ReportItem("Ta", self.Ta, "Período de vibración", "s",
                       _ref("6.3.3", "Determinación del período de vibración")),
            ReportItem("Sa(Ta)", self.Sa_Ta, "Aceleración espectral en Ta", "g",
                       _ref("3.3.1", "Espectro elástico de diseño")),
            ReportItem("I", self.I, "Coeficiente de importancia", "",
                       _ref("4.1", "Factor de importancia", table="6")),
            ReportItem("R", self.R, "Factor de reducción de resistencia sísmica", "",
                       _ref("6.3.4", "Factor R", table="15/16")),
            ReportItem("phi_P", self.phi_p, "Coeficiente de regularidad en planta", "",
                       _ref("5.3", "Coeficientes de configuración estructural")),
            ReportItem("phi_E", self.phi_e,
                       "Coeficiente de regularidad en elevación", "",
                       _ref("5.3", "Coeficientes de configuración estructural")),
            ReportItem("W", self.W, "Carga sísmica reactiva", "",
                       _ref("6.1.7", "Carga sísmica reactiva")),
            ReportItem("V", self.V, "I Sa(Ta) W / (R phi_P phi_E)", "",
                       _ref("6.3.2", "Cortante basal total de diseño")),
            ReportItem("V/W", self.shear_coefficient, "Coeficiente de cortante basal"),
            ReportItem("k", self.k, "Exponente de distribución vertical", "",
                       _ref("6.3.5", "Distribución vertical de fuerzas")),
        ]
        return Report(
            title="NEC-SE-DS 2015 cortante basal (DBF, §6.3)",
            items=tuple(items),
        )

    def __str__(self) -> str:
        return self.report().to_text()


def base_shear(
    spectrum: Spectrum,
    *,
    W: float,
    R: float,
    I: float = 1.0,
    phi_p: float = 1.0,
    phi_e: float = 1.0,
    T: float | None = None,
    hn: float | None = None,
    structure_type: str = "hormigon_porticos",
    weights: ArrayLike | None = None,
    heights: ArrayLike | None = None,
) -> NECELFResult:
    """Design base shear ``V = I Sa(Ta) W / (R phi_P phi_E)`` (§6.3.2).

    Parameters
    ----------
    spectrum
        The **elastic** spectrum from
        :meth:`~codeSpectra.codes.nec.NECSEDS2015.elastic_spectrum`. Pass the
        elastic curve, not the inelastic one — the reduction is applied here,
        and passing an already-reduced spectrum divides by ``R`` twice.
    W
        Carga sísmica reactiva (§6.1.7).
    T, hn
        Supply the period directly, or the height to compute ``Ta`` by Método
        1. If both are given, ``T`` is capped at 1.30 ``Ta`` per §6.3.3b.
    weights, heights
        Supply both to also return the vertical force distribution.
    """
    if W <= 0.0:
        raise InvalidInput("W must be positive.")
    if R <= 0.0:
        raise InvalidInput("R must be positive.")
    for name, value in (("I", I), ("phi_p", phi_p), ("phi_e", phi_e)):
        if value <= 0.0:
            raise InvalidInput(f"{name} must be positive, got {value}.")
    if T is None and hn is None:
        raise InvalidInput("Supply the period T, the height hn, or both.")

    Ta_1 = approximate_period(hn, structure_type) if hn is not None else None
    if T is None:
        assert Ta_1 is not None
        Ta = Ta_1
    elif Ta_1 is not None:
        Ta = limit_period_method_2(T, Ta_1)
    else:
        Ta = T

    Sa_Ta = float(spectrum.at(Ta))
    V = I * Sa_Ta * W / (R * phi_p * phi_e)
    k = vertical_distribution_exponent(Ta)

    Fx = Vx = None
    if weights is not None and heights is not None:
        Fx, Vx = vertical_distribution(V, weights, heights, Ta)

    return NECELFResult(
        V=V, Sa_Ta=Sa_Ta, Ta=Ta, k=k, R=R,
        phi_p=phi_p, phi_e=phi_e, I=I, W=W, Fx=Fx, Vx=Vx,
    )
