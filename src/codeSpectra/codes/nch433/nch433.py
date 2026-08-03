"""NCh433.Of1996 Mod.2009 with DS 61 (2011) — Chile.

Elastic modal spectrum (§6.3.5)::

    alpha(Tn) = [1 + 4.5 (Tn/To)**p] / [1 + (Tn/To)**3]        (Eq. 6-9)
    Sa(Tn)    = I * S * Ao * alpha(Tn)

Reduced by the period-dependent response modification factor (Eq. 6-10)::

    R*(T*) = 1 + T* / (0.10 To + T*/Ro)

``R*`` depends on ``T*``, the period of the mode with the greatest equivalent
translational mass in the direction of analysis — a single scalar per
direction, not a function evaluated at every period. That is why
:meth:`NCh433.reduction_factor` takes ``T_star`` explicitly and
:meth:`NCh433.design_spectrum` divides the whole curve by that one scalar.

Table values verified against NCh433.Of1996 Mod.2009 (Tablas 6.1-6.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.control import ControlPeriods
from ...core.exceptions import InvalidInput
from ...core.references import ClauseRef
from ...core.reports import Report, ReportItem
from ...core.spectrum import AnalyticSpectrum, SpectrumKind, SpectrumMeta

__all__ = ["BuildingCategory", "NCh433", "SeismicZone", "SoilType"]

STANDARD = "NCh433"
EDITION = "Of1996 Mod.2009 (DS 61-2011)"


def _ref(clause: str, description: str, **kw: str) -> ClauseRef:
    return ClauseRef(standard=STANDARD, edition=EDITION, clause=clause,
                     description=description, **kw)


#: Tabla 6.2 — effective acceleration Ao, as a fraction of g.
ZONE_AO: dict[str, float] = {"1": 0.20, "2": 0.30, "3": 0.40}

#: Tabla 6.1 — importance coefficient I by building category.
CATEGORY_I: dict[str, float] = {"I": 0.60, "II": 1.00, "III": 1.20, "IV": 1.20}

#: Tabla 6.3 — soil-dependent parameters: S, To, T', n, p.
SOIL_PARAMETERS: dict[str, dict[str, float]] = {
    "A": {"S": 0.90, "To": 0.15, "Tp": 0.20, "n": 1.00, "p": 2.0},
    "B": {"S": 1.00, "To": 0.30, "Tp": 0.35, "n": 1.33, "p": 1.5},
    "C": {"S": 1.05, "To": 0.40, "Tp": 0.45, "n": 1.40, "p": 1.6},
    "D": {"S": 1.20, "To": 0.75, "Tp": 0.85, "n": 1.80, "p": 1.0},
    "E": {"S": 1.30, "To": 1.20, "Tp": 1.35, "n": 1.80, "p": 1.0},
}

SOIL_DESCRIPTIONS: dict[str, str] = {
    "A": "Roca, suelo cementado",
    "B": "Roca blanda o fracturada, suelo muy denso o muy firme",
    "C": "Suelo denso o firme",
    "D": "Suelo medianamente denso, o firme",
    "E": "Suelo de compacidad, o consistencia mediana",
}

#: Tabla 6.4 — maximum seismic coefficient Cmax, as a multiple of S*Ao/g.
CMAX_BY_R: tuple[tuple[float, float], ...] = (
    (2.0, 0.90),
    (3.0, 0.60),
    (4.0, 0.55),
    (5.5, 0.40),
    (6.0, 0.35),
    (7.0, 0.35),
)


class SeismicZone(str, Enum):
    """Seismic zone 1-3 (Tabla 6.2)."""

    Z1 = "1"
    Z2 = "2"
    Z3 = "3"

    @property
    def Ao(self) -> float:
        """Effective acceleration as a fraction of g."""
        return ZONE_AO[self.value]


class SoilType(str, Enum):
    """Soil classification A-F (Tabla 4.2)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class BuildingCategory(str, Enum):
    """Building occupancy category I-IV (Tabla 4.3 / 6.1)."""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"

    @property
    def importance(self) -> float:
        """Importance coefficient I."""
        return CATEGORY_I[self.value]


@dataclass(frozen=True)
class NCh433:
    """Seismic design spectrum per NCh433 (Chile).

    Examples
    --------
    >>> site = NCh433(zone="2", soil="B", category="II")
    >>> site.Ao, site.S, site.To
    (0.3, 1.0, 0.3)
    >>> round(site.elastic_spectrum().at(0.0), 4)
    0.3
    """

    zone: SeismicZone | str = SeismicZone.Z2
    soil: SoilType | str = SoilType.B
    category: BuildingCategory | str = BuildingCategory.II

    def __post_init__(self) -> None:
        object.__setattr__(self, "zone", SeismicZone(self.zone))
        object.__setattr__(self, "soil", SoilType(self.soil))
        object.__setattr__(self, "category", BuildingCategory(self.category))
        if self.soil_type == "F":
            raise InvalidInput(
                "Soil type F requires a site-specific study; NCh433 Tabla 6.3 "
                "tabulates no parameters for it."
            )

    # -- Accessors ---------------------------------------------------------
    @property
    def soil_type(self) -> str:
        """Soil classification as a plain string."""
        return SoilType(self.soil).value

    @property
    def Ao(self) -> float:
        """Effective acceleration, fraction of g (Tabla 6.2)."""
        return SeismicZone(self.zone).Ao

    @property
    def I(self) -> float:  # noqa: E743
        """Importance coefficient (Tabla 6.1)."""
        return BuildingCategory(self.category).importance

    @cached_property
    def _soil(self) -> dict[str, float]:
        return SOIL_PARAMETERS[self.soil_type]

    @property
    def S(self) -> float:
        """Soil amplification factor (Tabla 6.3)."""
        return self._soil["S"]

    @property
    def To(self) -> float:
        """Soil characteristic period, s (Tabla 6.3)."""
        return self._soil["To"]

    @property
    def Tp(self) -> float:
        """Second soil characteristic period T', s (Tabla 6.3)."""
        return self._soil["Tp"]

    @property
    def n(self) -> float:
        """Soil exponent n (Tabla 6.3)."""
        return self._soil["n"]

    @property
    def p(self) -> float:
        """Soil exponent p (Tabla 6.3)."""
        return self._soil["p"]

    @property
    def description(self) -> str:
        """Soil description (Tabla 4.2)."""
        return SOIL_DESCRIPTIONS[self.soil_type]

    @property
    def control_periods(self) -> ControlPeriods:
        """The soil characteristic periods ``To`` and ``T'``."""
        return ControlPeriods(To=self.To, Tp=self.Tp)

    # -- Spectrum shape ------------------------------------------------------
    def alpha(self, T: ArrayLike) -> NDArray[np.float64]:
        """Amplification factor ``alpha(Tn)`` (Eq. 6-9)."""
        Tn = np.atleast_1d(np.asarray(T, dtype=float))
        ratio = Tn / self.To
        return (1.0 + 4.5 * ratio**self.p) / (1.0 + ratio**3)

    def elastic_spectrum(self, *, t_max: float = 4.0) -> AnalyticSpectrum:
        """Elastic modal spectrum ``Sa = I S Ao alpha(Tn)``, in g (§6.3.5)."""
        I, S, Ao = self.I, self.S, self.Ao

        def evaluate(T: NDArray[np.float64]) -> NDArray[np.float64]:
            return I * S * Ao * self.alpha(T)

        meta = SpectrumMeta(
            standard=STANDARD,
            edition=EDITION,
            kind=SpectrumKind.ELASTIC,
            label=f"NCh433 elastic spectrum (Zona {SeismicZone(self.zone).value}, "
                  f"suelo {self.soil_type})",
            control_periods=self.control_periods,
            parameters=self._parameters(),
            refs=(_ref("6.3.5", "Espectro de diseno", equation="6-9"),),
        )
        return AnalyticSpectrum(evaluate, meta=meta, t_max=t_max)

    def reduction_factor(self, T_star: float, Ro: float) -> float:
        """Period-dependent reduction factor ``R*`` (Eq. 6-10).

        Parameters
        ----------
        T_star
            Period of the mode with the greatest equivalent translational mass
            in the direction of analysis, s.
        Ro
            Base response modification factor (Tabla 5.1).
        """
        if T_star <= 0.0:
            raise InvalidInput("T* must be positive.")
        if Ro <= 0.0:
            raise InvalidInput("Ro must be positive.")
        return 1.0 + T_star / (0.10 * self.To + T_star / Ro)

    def design_spectrum(
        self, T_star: float, Ro: float, *, t_max: float = 4.0
    ) -> AnalyticSpectrum:
        """Elastic spectrum divided by ``R*(T*)`` (§6.3.5).

        ``R*`` is a single scalar per direction of analysis, so the entire
        curve is scaled by it rather than reduced period-by-period.
        """
        R_star = self.reduction_factor(T_star, Ro)
        reduced = self.elastic_spectrum(t_max=t_max).reduced(
            R_star,
            label=f"NCh433 design spectrum (R*={R_star:.3f}, T*={T_star:g} s, "
                  f"Ro={Ro:g})",
        )
        assert isinstance(reduced, AnalyticSpectrum)
        return reduced

    # -- Base shear coefficient limits ---------------------------------------
    @property
    def C_min(self) -> float:
        """Minimum seismic coefficient ``S Ao / (6 g)`` (§6.3.7.1)."""
        return self.S * self.Ao / 6.0

    def C_max(self, R: float) -> float:
        """Maximum seismic coefficient (Tabla 6.4), as a fraction of g.

        Straight-line interpolation between the tabulated ``R`` values, held
        constant outside the tabulated range.
        """
        if R <= 0.0:
            raise InvalidInput("R must be positive.")
        xs = np.array([row[0] for row in CMAX_BY_R])
        ys = np.array([row[1] for row in CMAX_BY_R])
        return float(np.interp(R, xs, ys)) * self.S * self.Ao

    def Sa_min(self) -> float:
        """Minimum design spectral acceleration ``I S Ao / 6``, in g."""
        return self.I * self.C_min

    # -- Reporting ------------------------------------------------------------
    def _parameters(self) -> dict[str, float | str]:
        return {
            "zone": SeismicZone(self.zone).value,
            "soil": self.soil_type,
            "Ao": self.Ao,
            "I": self.I,
            "S": self.S,
            "To": self.To,
            "Tp": self.Tp,
            "n": self.n,
            "p": self.p,
        }

    def report(self) -> Report:
        """A citation-carrying record of the parameter derivation."""
        items = [
            ReportItem("Zona", SeismicZone(self.zone).value, "Zona sismica", "",
                       _ref("6.2.3.2", "Zonificacion sismica", table="6.2")),
            ReportItem("Ao", self.Ao, "Aceleracion efectiva maxima", "g",
                       _ref("6.2.3.2", "Aceleracion efectiva", table="6.2")),
            ReportItem("Categoria", BuildingCategory(self.category).value,
                       "Categoria del edificio", "",
                       _ref("4.3", "Categoria de ocupacion", table="4.3")),
            ReportItem("I", self.I, "Coeficiente de importancia", "",
                       _ref("6.3.5", "Coeficiente I", table="6.1")),
            ReportItem("Suelo", self.soil_type, self.description, "",
                       _ref("4.2", "Clasificacion sismica del terreno", table="4.2")),
            ReportItem("S", self.S, "Factor de amplificacion del suelo", "",
                       _ref("6.3.5", "Parametros del suelo", table="6.3")),
            ReportItem("To", self.To, "Periodo caracteristico del suelo", "s",
                       _ref("6.3.5", "Parametros del suelo", table="6.3")),
            ReportItem("T'", self.Tp, "Segundo periodo caracteristico", "s",
                       _ref("6.3.5", "Parametros del suelo", table="6.3")),
            ReportItem("n", self.n, "Exponente n", "",
                       _ref("6.3.5", "Parametros del suelo", table="6.3")),
            ReportItem("p", self.p, "Exponente p", "",
                       _ref("6.3.5", "Parametros del suelo", table="6.3")),
            ReportItem("C_min", self.C_min, "S Ao / 6", "g",
                       _ref("6.3.7.1", "Coeficiente sismico minimo")),
        ]
        return Report(
            title=f"NCh433 — Zona {SeismicZone(self.zone).value}, "
                  f"suelo {self.soil_type}",
            items=tuple(items),
            notes=(
                "Eq. 6-11, the alternative R* for wall-type buildings, is not "
                "implemented; use reduction_factor() (Eq. 6-10).",
            ),
        )

    def __str__(self) -> str:
        return self.report().to_text()
