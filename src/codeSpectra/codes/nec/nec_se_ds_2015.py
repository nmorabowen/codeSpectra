"""NEC-SE-DS 2015 — Peligro Sísmico y Diseño Sismo Resistente (Ecuador).

Elastic spectrum (§3.3.1)::

    Sa = eta * Z * Fa                        for 0 <= T <= Tc
    Sa = eta * Z * Fa * (Tc / T) ** r        for T > Tc

with ``Tc = 0.55 Fs Fd / Fa`` and ``T0 = 0.10 Fs Fd / Fa``.

The ascending branch below ``T0``::

    Sa = Z * Fa * [1 + (eta - 1) * T / T0]

is **not** part of the design spectrum. NEC-SE-DS restricts it to dynamic
analysis, and within that only to modes other than the fundamental one. It is
therefore off by default; pass ``include_ascending_branch=True`` to build the
curve used for higher-mode evaluation. Getting this backwards under-estimates
short-period demand, so the two curves are deliberately separate calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from ...core.control import ControlPeriods
from ...core.exceptions import InvalidInput
from ...core.references import ClauseRef
from ...core.reports import Report, ReportItem
from ...core.spectrum import AnalyticSpectrum, SpectrumKind, SpectrumMeta
from . import _tables as NT

__all__ = [
    "NECSEDS2015",
    "OccupancyCategory",
    "Region",
    "SeismicZone",
    "SoilType",
]

STANDARD = NT.STANDARD
EDITION = NT.EDITION


def _ref(clause: str, description: str, **kw: str) -> ClauseRef:
    return ClauseRef(standard=STANDARD, edition=EDITION, clause=clause,
                     description=description, **kw)


class SeismicZone(str, Enum):
    """Seismic zone I-VI (Tabla 1, Figura 1)."""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"
    V = "V"
    VI = "VI"

    @property
    def Z(self) -> float:
        """Design PGA on rock as a fraction of g."""
        return NT.ZONE_Z[self.value]

    @property
    def hazard(self) -> str:
        """Qualitative hazard characterisation (Tabla 1)."""
        return NT.ZONE_HAZARD[self.value]


class SoilType(str, Enum):
    """Soil profile type A-F (Tabla 2)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class Region(str, Enum):
    """Geographic region controlling the spectral amplification ratio eta.

    Per §3.3.1: 1.80 for the Costa provinces excluding Esmeraldas; 2.48 for
    the Sierra provinces plus Esmeraldas and Galápagos; 2.60 for the Oriente.
    Esmeraldas and Galápagos take the Sierra value.
    """

    COSTA = "costa"
    SIERRA = "sierra"
    ORIENTE = "oriente"

    @property
    def eta(self) -> float:
        """Ratio Sa(T = 0.1 s) / PGA for the design return period."""
        return {"costa": 1.80, "sierra": 2.48, "oriente": 2.60}[self.value]


class OccupancyCategory(str, Enum):
    """Occupancy category driving the importance coefficient I (Tabla 6)."""

    ESENCIAL = "esencial"
    ESPECIAL = "especial"
    OTRA = "otra"

    @property
    def I(self) -> float:  # noqa: E743
        """Importance coefficient."""
        return NT.IMPORTANCE[self.value]


@dataclass(frozen=True)
class NECSEDS2015:
    """Seismic hazard parameters and spectra per NEC-SE-DS 2015.

    Parameters
    ----------
    zone
        Seismic zone I-VI. Supply this or ``Z``, not both.
    Z
        Design PGA on rock in g, if taken from Tabla 19 (poblaciones) rather
        than from the zone map. Values between tabulated columns are
        interpolated, which the standard does not explicitly sanction; a note
        is added to the report when this happens.
    soil
        Soil profile type A-F (Tabla 2). Type F raises, since NEC requires a
        site-specific spectrum.
    region
        Geographic region setting ``eta``.
    occupancy
        Occupancy category setting the importance coefficient ``I``.
    Fa_override, Fd_override, Fs_override
        Explicit soil coefficient overrides, for a completed site-specific
        study.

    Examples
    --------
    >>> site = NECSEDS2015(zone="V", soil="D", region="sierra")
    >>> site.Z_g, site.Fa, site.Fd, site.Fs
    (0.4, 1.2, 1.19, 1.28)
    >>> round(site.Sa_plateau, 4)   # eta * Z * Fa
    1.1904
    """

    zone: SeismicZone | str | None = None
    soil: SoilType | str = SoilType.D
    region: Region | str = Region.SIERRA
    Z: float | None = None
    occupancy: OccupancyCategory | str = OccupancyCategory.OTRA
    Fa_override: float | None = None
    Fd_override: float | None = None
    Fs_override: float | None = None
    provenance_note: str | None = None

    def __post_init__(self) -> None:
        if self.zone is None and self.Z is None:
            raise InvalidInput("Supply either zone (I-VI) or Z.")
        if self.zone is not None:
            object.__setattr__(self, "zone", SeismicZone(self.zone))
            zone_Z = SeismicZone(self.zone).Z
            if self.Z is not None and abs(self.Z - zone_Z) > 1e-12:
                raise InvalidInput(
                    f"zone {SeismicZone(self.zone).value} implies Z = {zone_Z:g}, "
                    f"but Z = {self.Z:g} was given. Supply only one."
                )
            object.__setattr__(self, "Z", zone_Z)
        object.__setattr__(self, "soil", SoilType(self.soil))
        object.__setattr__(self, "region", Region(self.region))
        object.__setattr__(self, "occupancy", OccupancyCategory(self.occupancy))
        assert self.Z is not None
        if not 0.0 < self.Z <= 1.0:
            raise InvalidInput(f"Z must lie in (0, 1] g, got {self.Z:g}.")

    # -- Convenience accessors ---------------------------------------------
    @property
    def Z_g(self) -> float:
        """Design PGA on rock in g, guaranteed resolved (from zone or input)."""
        assert self.Z is not None
        return float(self.Z)

    @property
    def soil_type(self) -> str:
        """Soil profile type as a plain string."""
        return SoilType(self.soil).value

    @property
    def eta(self) -> float:
        """Spectral amplification ratio Sa(0.1 s)/PGA for the region."""
        return Region(self.region).eta

    @property
    def I(self) -> float:  # noqa: E743
        """Importance coefficient (Tabla 6)."""
        return OccupancyCategory(self.occupancy).I

    @property
    def r(self) -> float:
        """Decay exponent: 1.0 for every soil except type E, where it is 1.5."""
        return 1.5 if self.soil_type == "E" else 1.0

    # -- Soil coefficients --------------------------------------------------
    @cached_property
    def soil_coefficients(self) -> tuple[float, float, float]:
        """``(Fa, Fd, Fs)`` from Tablas 3, 4 and 5."""
        soil, Z = self.soil_type, self.Z_g
        Fa = self.Fa_override if self.Fa_override is not None else NT.FA_TABLE.lookup(soil, Z)
        Fd = self.Fd_override if self.Fd_override is not None else NT.FD_TABLE.lookup(soil, Z)
        Fs = self.Fs_override if self.Fs_override is not None else NT.FS_TABLE.lookup(soil, Z)
        return Fa, Fd, Fs

    @property
    def Fa(self) -> float:
        """Short-period soil amplification actually used (Tabla 3)."""
        return self.soil_coefficients[0]

    @property
    def Fd(self) -> float:
        """Displacement-spectrum soil amplification actually used (Tabla 4)."""
        return self.soil_coefficients[1]

    @property
    def Fs(self) -> float:
        """Nonlinear soil behaviour factor actually used (Tabla 5)."""
        return self.soil_coefficients[2]

    # -- Control periods ----------------------------------------------------
    @cached_property
    def control_periods(self) -> ControlPeriods:
        """``T0``, ``Tc`` and ``TL`` per §3.3.1.

        ``TL = 2.4 Fd``, capped at 4 s for soil profile types D and E.
        """
        Fa, Fd, Fs = self.soil_coefficients
        ratio = Fs * Fd / Fa
        TL = 2.4 * Fd
        if self.soil_type in ("D", "E"):
            TL = min(TL, 4.0)
        return ControlPeriods(T0=0.10 * ratio, Tc=0.55 * ratio, TL=TL)

    @property
    def Sa_plateau(self) -> float:
        """Plateau ordinate ``eta * Z * Fa``, in g."""
        return self.eta * self.Z_g * self.Fa

    # -- Spectra -------------------------------------------------------------
    def elastic_spectrum(
        self,
        *,
        include_ascending_branch: bool = False,
        t_max: float = 4.0,
    ) -> AnalyticSpectrum:
        """The §3.3.1 elastic acceleration spectrum, in g.

        Parameters
        ----------
        include_ascending_branch
            Include the ``T <= T0`` ramp ``Z Fa [1 + (eta-1) T/T0]``. NEC
            permits this **only** in dynamic analysis, and only for modes
            other than the fundamental one. Leave False for design.
        t_max
            Upper period of the reported range, s.
        """
        cp = self.control_periods
        Fa = self.Fa
        Z, eta, r = self.Z_g, self.eta, self.r
        plateau = eta * Z * Fa
        Tc, T0 = cp.Tc, cp.T0

        def evaluate(T: NDArray[np.float64]) -> NDArray[np.float64]:
            T = np.asarray(T, dtype=float)
            out = np.full(T.shape, plateau, dtype=float)
            decay = Tc < T
            safe = np.where(T > 0.0, T, 1.0)
            out[decay] = plateau * (Tc / safe[decay]) ** r
            if include_ascending_branch and T0 > 0.0:
                ramp = T <= T0
                out[ramp] = Z * Fa * (1.0 + (eta - 1.0) * T[ramp] / T0)
            return out

        branch = " (with ascending branch)" if include_ascending_branch else ""
        refs = [_ref("3.3.1", "Espectro elástico horizontal de diseño en aceleraciones")]
        if include_ascending_branch:
            refs.append(
                _ref("3.3.1", "Rama ascendente para modos distintos al fundamental, "
                              "solo en análisis dinámico")
            )
        meta = SpectrumMeta(
            standard=STANDARD,
            edition=EDITION,
            kind=SpectrumKind.ELASTIC,
            damping=0.05,
            label=f"NEC-SE-DS 2015 elastic spectrum "
                  f"(Z={Z:g}, soil {self.soil_type}){branch}",
            control_periods=cp,
            parameters=self._parameters(),
            refs=tuple(refs),
        )
        return AnalyticSpectrum(evaluate, meta=meta, t_max=t_max)

    def inelastic_spectrum(
        self,
        R: float,
        *,
        phi_p: float = 1.0,
        phi_e: float = 1.0,
        apply_importance: bool = True,
        include_ascending_branch: bool = False,
        t_max: float = 4.0,
    ) -> AnalyticSpectrum:
        """Design spectrum reduced per §6.3.2: ``I Sa / (R phi_P phi_E)``.

        Parameters
        ----------
        R
            Seismic response reduction factor (Tabla 15 or 16).
        phi_p, phi_e
            Plan and elevation configuration coefficients (§5.3). Values below
            1.0 penalise irregular structures by increasing demand.
        apply_importance
            Multiply by the importance coefficient ``I``. True matches the
            base-shear equation of §6.3.2.
        """
        if R <= 0.0:
            raise InvalidInput(f"R must be positive, got {R}.")
        for name, value in (("phi_p", phi_p), ("phi_e", phi_e)):
            if not 0.0 < value <= 1.0:
                raise InvalidInput(f"{name} must lie in (0, 1], got {value}.")
        base = self.elastic_spectrum(
            include_ascending_branch=include_ascending_branch, t_max=t_max
        )
        Ie = self.I if apply_importance else 1.0
        reduced = base.reduced(
            R,
            Ie=Ie,
            phi_p=phi_p,
            phi_e=phi_e,
            label=f"NEC-SE-DS 2015 inelastic spectrum (R={R:g}, "
                  f"phiP={phi_p:g}, phiE={phi_e:g})",
        )
        assert isinstance(reduced, AnalyticSpectrum)
        return reduced

    def displacement_spectrum(self, *, t_max: float | None = None) -> AnalyticSpectrum:
        """Elastic displacement spectrum ``Sd`` in metres (§3.3.2).

        Reported to ``TL``; beyond that the standard holds the displacement
        constant at its ``TL`` value.
        """
        from ...core.units import STANDARD_GRAVITY

        cp = self.control_periods
        TL = cp.TL
        accel = self.elastic_spectrum(t_max=max(TL, 4.0))

        def evaluate(T: NDArray[np.float64]) -> NDArray[np.float64]:
            T = np.asarray(T, dtype=float)
            clipped = np.minimum(T, TL)
            Sa = np.atleast_1d(accel.at(clipped))
            Sd = Sa * STANDARD_GRAVITY * (clipped / (2.0 * np.pi)) ** 2
            return np.asarray(Sd, dtype=np.float64)

        meta = SpectrumMeta(
            standard=STANDARD,
            edition=EDITION,
            kind=SpectrumKind.DISPLACEMENT,
            label=f"NEC-SE-DS 2015 displacement spectrum (Z={self.Z_g:g}, "
                  f"soil {self.soil_type})",
            control_periods=cp,
            parameters=self._parameters(),
            refs=(_ref("3.3.2", "Espectro elástico de desplazamientos"),),
        )
        return AnalyticSpectrum(evaluate, meta=meta, t_max=t_max or max(TL, 4.0))

    # -- Reporting -----------------------------------------------------------
    def _parameters(self) -> dict[str, float | str]:
        Fa, Fd, Fs = self.soil_coefficients
        return {
            "Z": self.Z_g,
            "soil": self.soil_type,
            "region": Region(self.region).value,
            "eta": self.eta,
            "r": self.r,
            "I": self.I,
            "Fa": Fa,
            "Fd": Fd,
            "Fs": Fs,
        }

    @property
    def _interpolated_Z(self) -> bool:
        """Whether Z falls between the tabulated columns of Tablas 3-5."""
        return not any(abs(self.Z_g - c) < 1e-12 for c in NT.FA_TABLE.columns)

    def report(self) -> Report:
        """A citation-carrying record of the full parameter derivation."""
        cp = self.control_periods
        Fa, Fd, Fs = self.soil_coefficients
        zone_label = SeismicZone(self.zone).value if self.zone is not None else "—"
        items = [
            ReportItem("Zona", zone_label, "Zona sísmica", "",
                       _ref("3.1.1", "Zonificación sísmica", table="1")),
            ReportItem("Z", self.Z_g, "Aceleración máxima en roca", "g",
                       _ref("3.1.1", "Factor de zona", table="1")),
            ReportItem("Suelo", self.soil_type, "Tipo de perfil de subsuelo", "",
                       _ref("3.2.1", "Clasificación de los perfiles de suelo",
                            table="2")),
            ReportItem("Región", Region(self.region).value, "Región geográfica"),
            ReportItem("eta", self.eta, "Razón Sa(0.1 s)/PGA", "",
                       _ref("3.3.1", "Relación de amplificación espectral")),
            ReportItem("r", self.r, "Exponente de decaimiento", "",
                       _ref("3.3.1", "Factor r del espectro elástico")),
            ReportItem("I", self.I, "Coeficiente de importancia", "",
                       _ref("4.1", "Factor de importancia", table="6")),
            ReportItem("Fa", Fa, "Amplificación en período corto", "",
                       NT.FA_TABLE.ref),
            ReportItem("Fd", Fd, "Amplificación del espectro de desplazamientos", "",
                       NT.FD_TABLE.ref),
            ReportItem("Fs", Fs, "Comportamiento inelástico del subsuelo", "",
                       NT.FS_TABLE.ref),
            ReportItem("T0", cp.T0, "0.10 Fs Fd / Fa", "s",
                       _ref("3.3.1", "Período límite inferior")),
            ReportItem("Tc", cp.Tc, "0.55 Fs Fd / Fa", "s",
                       _ref("3.3.1", "Período límite del plateau")),
            ReportItem("TL", cp.TL, "2.4 Fd (<= 4 s para suelos D y E)", "s",
                       _ref("3.3.1", "Período límite para desplazamientos")),
            ReportItem("Sa(plateau)", self.Sa_plateau, "eta Z Fa", "g",
                       _ref("3.3.1", "Ordenada del plateau")),
        ]
        notes: list[str] = []
        # Non-code provenance leads: a reader must see it before the numbers.
        if self.provenance_note:
            notes.append(self.provenance_note)
        if self._interpolated_Z:
            notes.append(
                f"Z = {self.Z_g:g} g falls between the tabulated columns of Tablas "
                "3-5; Fa, Fd and Fs were obtained by straight-line interpolation. "
                "NEC-SE-DS tabulates these against the discrete zone values and "
                "does not state an interpolation rule — confirm this is acceptable."
            )
        if self.soil_type in ("D", "E") and 2.4 * Fd > 4.0:
            notes.append(
                f"TL capped at 4.0 s (2.4 Fd = {2.4 * Fd:.2f} s) for soil profile "
                "type " + self.soil_type + " per the §3.3.1 note."
            )
        return Report(
            title=f"NEC-SE-DS 2015 — Zona {zone_label}, suelo {self.soil_type}, "
                  f"{Region(self.region).value}",
            items=tuple(items),
            notes=tuple(notes),
        )

    def __str__(self) -> str:
        return self.report().to_text()
