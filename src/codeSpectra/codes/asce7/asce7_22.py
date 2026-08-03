"""ASCE/SEI 7-22 seismic ground motion (Chapter 11).

7-22 changed the input contract. Per §11.4.3, ``Ss``, ``S1``, ``SMS`` and
``SM1`` are all read **directly from the USGS Seismic Design Geodatabase for
the applicable site class** — the ``Fa``/``Fv`` site-coefficient tables of
7-16 were removed. There is therefore no way to go from mapped ``Ss``/``S1``
plus a site class to ``SMS``/``SM1`` inside this library, and this module does
not pretend otherwise: you supply site-adjusted values.

Two construction paths mirror the two the standard defines:

:meth:`ASCE7_22.from_mprs`
    §11.4.5.1, the default. You supply the 22 multi-period 5%-damped MCEr
    ordinates; the design spectrum is 2/3 of them, linearly interpolated.

:meth:`ASCE7_22.from_site_adjusted`
    §11.4.5.2, permitted by §11.4.5 Exception 2 only where multi-period values
    are unavailable. You supply ``SMS`` and ``SM1``.

The two are **not** interchangeable for long-period or irregular structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from ...core.control import ControlPeriods
from ...core.exceptions import InvalidInput
from ...core.reports import Report, ReportItem
from ...core.spectrum import (
    AnalyticSpectrum,
    Spectrum,
    SpectrumKind,
    SpectrumMeta,
    TabulatedSpectrum,
)
from ._shared import (
    STANDARD,
    RiskCategory,
    SeismicDesignCategory,
    SiteClass,
    ref,
    seismic_design_category,
    two_period_spectrum,
)

__all__ = ["ASCE7_22", "MPRS_PERIODS", "SpectrumBasis"]

EDITION = "7-22"

#: §11.4.5.1 item 1 — the 22 discrete periods the geodatabase tabulates, in s.
MPRS_PERIODS: tuple[float, ...] = (
    0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
)

#: Site classes 7-22 defines (§11.4.2, Table 20.2-1).
PERMITTED_SITE_CLASSES = ("A", "B", "BC", "C", "CD", "D", "DE", "E", "F")


class SpectrumBasis(str, Enum):
    """Which §11.4.5 path produced the spectrum."""

    MULTI_PERIOD = "multi-period"
    """§11.4.5.1, from geodatabase multi-period ordinates. The default."""

    TWO_PERIOD = "two-period"
    """§11.4.5.2, permitted only under §11.4.5 Exception 2."""


@dataclass(frozen=True)
class ASCE7_22:
    """Seismic ground motion parameters and spectra per ASCE/SEI 7-22.

    Build with :meth:`from_mprs` or :meth:`from_site_adjusted` rather than
    calling the constructor directly — the two paths take different inputs.

    Examples
    --------
    >>> site = ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD")
    >>> round(site.SDS, 4), round(site.SD1, 4)
    (1.0, 0.68)
    >>> site.basis
    <SpectrumBasis.TWO_PERIOD: 'two-period'>
    """

    site_class: SiteClass | str = SiteClass.CD
    TL: float = 8.0
    risk_category: RiskCategory | str = RiskCategory.II
    basis: SpectrumBasis = SpectrumBasis.MULTI_PERIOD
    SMS: float = 0.0
    SM1: float = 0.0
    Ss: float | None = None
    S1: float | None = None
    mprs_periods: tuple[float, ...] = ()
    mprs_mcer: tuple[float, ...] = ()
    default_site_conditions: bool = False
    _SDS_input: float | None = field(default=None, repr=False)
    _SD1_input: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_class", SiteClass(self.site_class))
        object.__setattr__(self, "risk_category", RiskCategory(self.risk_category))
        if self.sc not in PERMITTED_SITE_CLASSES:
            raise InvalidInput(
                f"ASCE 7-22 defines Site Classes {PERMITTED_SITE_CLASSES}; "
                f"got {self.sc!r}."
            )
        if self.TL <= 0.0:
            raise InvalidInput("TL must be positive.")

    # -- Constructors ------------------------------------------------------
    @classmethod
    def from_mprs(
        cls,
        periods: ArrayLike,
        sa_mcer: ArrayLike,
        *,
        site_class: SiteClass | str = SiteClass.CD,
        TL: float = 8.0,
        risk_category: RiskCategory | str = RiskCategory.II,
        SMS: float | None = None,
        SM1: float | None = None,
        Ss: float | None = None,
        S1: float | None = None,
        default_site_conditions: bool = False,
    ) -> ASCE7_22:
        """Build from multi-period MCEr ordinates (§11.4.5.1, the default path).

        Parameters
        ----------
        periods, sa_mcer
            The 5%-damped MCEr response spectrum from the USGS Seismic Design
            Geodatabase for the applicable site class. ``periods`` is normally
            :data:`MPRS_PERIODS`; ``sa_mcer`` are MCEr (not design) ordinates
            in g. The design spectrum is 2/3 of these.
        SMS, SM1
            Site-adjusted MCEr parameters from the geodatabase (§11.4.3). If
            omitted they are inferred from the ordinates using the §21.4
            rules, and the report says so.
        """
        T = np.atleast_1d(np.asarray(periods, dtype=float))
        Sa = np.atleast_1d(np.asarray(sa_mcer, dtype=float))
        if T.size != Sa.size:
            raise InvalidInput(
                f"periods and sa_mcer length mismatch: {T.size} vs {Sa.size}."
            )
        if T.size < 2:
            raise InvalidInput("At least two multi-period ordinates are required.")
        if np.any(Sa < 0.0):
            raise InvalidInput("MCEr ordinates must be non-negative.")

        derived_SMS, derived_SD1_basis = _infer_parameters(T, Sa, site_class)
        return cls(
            site_class=site_class,
            TL=TL,
            risk_category=risk_category,
            basis=SpectrumBasis.MULTI_PERIOD,
            SMS=SMS if SMS is not None else derived_SMS,
            SM1=SM1 if SM1 is not None else derived_SD1_basis,
            Ss=Ss,
            S1=S1,
            mprs_periods=tuple(float(t) for t in T),
            mprs_mcer=tuple(float(s) for s in Sa),
            default_site_conditions=default_site_conditions,
        )

    @classmethod
    def from_site_adjusted(
        cls,
        *,
        SMS: float,
        SM1: float,
        site_class: SiteClass | str = SiteClass.CD,
        TL: float = 8.0,
        risk_category: RiskCategory | str = RiskCategory.II,
        Ss: float | None = None,
        S1: float | None = None,
        default_site_conditions: bool = False,
    ) -> ASCE7_22:
        """Build the two-period spectrum from geodatabase ``SMS``/``SM1``.

        Permitted by §11.4.5 Exception 2 only where multi-period values are
        unavailable from the geodatabase. ``SMS`` and ``SM1`` come from
        §11.4.3 — they are *not* ``Fa*Ss`` and ``Fv*S1``, which 7-22 removed.
        """
        if SMS <= 0.0:
            raise InvalidInput("SMS must be positive.")
        if SM1 < 0.0:
            raise InvalidInput("SM1 must be non-negative.")
        return cls(
            site_class=site_class,
            TL=TL,
            risk_category=risk_category,
            basis=SpectrumBasis.TWO_PERIOD,
            SMS=SMS,
            SM1=SM1,
            Ss=Ss,
            S1=S1,
            default_site_conditions=default_site_conditions,
        )

    # -- Parameters ---------------------------------------------------------
    @property
    def sc(self) -> str:
        """Site class as a plain string."""
        return SiteClass(self.site_class).value

    @cached_property
    def SDS(self) -> float:
        """``(2/3) SMS`` (Eq. 11.4-1)."""
        return (2.0 / 3.0) * self.SMS

    @cached_property
    def SD1(self) -> float:
        """``(2/3) SM1`` (Eq. 11.4-2)."""
        return (2.0 / 3.0) * self.SM1

    @cached_property
    def control_periods(self) -> ControlPeriods:
        """``T0``, ``Ts`` and ``TL`` (§11.4.5.2)."""
        Ts = self.SD1 / self.SDS if self.SDS > 0.0 else 0.0
        return ControlPeriods(T0=0.2 * Ts, Ts=Ts, TL=self.TL)

    @property
    def Ie(self) -> float:
        """Seismic importance factor (Table 1.5-2)."""
        return RiskCategory(self.risk_category).importance_factor

    @cached_property
    def seismic_design_category(self) -> SeismicDesignCategory:
        """Seismic design category per §11.6.

        Uses ``S1`` where supplied for the ``S1 >= 0.75`` override; without it,
        the override cannot be evaluated and only Tables 11.6-1/11.6-2 apply.
        """
        return seismic_design_category(
            self.SDS, self.SD1, self.S1 if self.S1 is not None else 0.0,
            RiskCategory(self.risk_category),
        )

    # -- Spectra -------------------------------------------------------------
    def design_spectrum(self, *, t_max: float | None = None) -> Spectrum:
        """The §11.4.5 design response spectrum, by whichever basis applies."""
        if self.basis is SpectrumBasis.TWO_PERIOD:
            return two_period_spectrum(
                edition=EDITION,
                SDS=self.SDS,
                SD1=self.SD1,
                TL=self.TL,
                kind=SpectrumKind.DESIGN,
                label=f"ASCE 7-22 two-period design spectrum (Site Class {self.sc})",
                parameters=self._parameters(),
                refs=(
                    ref(EDITION, "11.4.5.2", "Two-period design response spectrum",
                        figure="11.4-1"),
                ),
                t_max=t_max,
            )

        design = np.asarray(self.mprs_mcer, dtype=float) * (2.0 / 3.0)
        meta = SpectrumMeta(
            standard=STANDARD,
            edition=EDITION,
            kind=SpectrumKind.DESIGN,
            damping=0.05,
            label=f"ASCE 7-22 multi-period design spectrum (Site Class {self.sc})",
            control_periods=ControlPeriods(**self.control_periods.as_dict()),
            parameters=self._parameters(),
            refs=(
                ref(EDITION, "11.4.5.1", "Multi-period design response spectrum"),
            ),
        )
        return TabulatedSpectrum(self.mprs_periods, design, meta=meta)

    def mcer_spectrum(self, *, t_max: float | None = None) -> Spectrum:
        """The MCEr response spectrum: 1.5x the design spectrum (§11.4.6)."""
        base = self.design_spectrum(t_max=t_max)
        return base.scaled(
            1.5, label=f"ASCE 7-22 MCEr spectrum (Site Class {self.sc})"
        )

    def two_period_spectrum(self, *, t_max: float | None = None) -> AnalyticSpectrum:
        """The §11.4.5.2 two-period spectrum, regardless of the active basis.

        Useful for showing how much the multi-period spectrum departs from the
        two-period idealisation at long periods.
        """
        return two_period_spectrum(
            edition=EDITION,
            SDS=self.SDS,
            SD1=self.SD1,
            TL=self.TL,
            kind=SpectrumKind.DESIGN,
            label=f"ASCE 7-22 two-period design spectrum (Site Class {self.sc})",
            parameters=self._parameters(),
            refs=(ref(EDITION, "11.4.5.2", "Two-period design response spectrum"),),
            t_max=t_max,
        )

    # -- Reporting ------------------------------------------------------------
    def _parameters(self) -> dict[str, float | str]:
        params: dict[str, float | str] = {
            "site_class": self.sc,
            "SMS": self.SMS,
            "SM1": self.SM1,
            "SDS": self.SDS,
            "SD1": self.SD1,
            "TL": self.TL,
            "basis": self.basis.value,
        }
        if self.Ss is not None:
            params["Ss"] = self.Ss
        if self.S1 is not None:
            params["S1"] = self.S1
        return params

    def report(self) -> Report:
        """A citation-carrying record of the parameter derivation."""
        cp = self.control_periods
        geodb = ref(EDITION, "11.4.3",
                    "MCEr parameters from the USGS Seismic Design Geodatabase")
        items = [
            ReportItem("Site Class", self.sc, "Chapter 20 site class", "",
                       ref(EDITION, "11.4.2", "Site class", table="20.2-1")),
            ReportItem("Basis", self.basis.value, "Design spectrum basis", "",
                       ref(EDITION, "11.4.5", "Design response spectrum")),
            ReportItem("Ie", self.Ie, "Seismic importance factor", "",
                       ref(EDITION, "11.5.1", table="1.5-2")),
            ReportItem("SMS", self.SMS, "Site-adjusted MCEr, short period", "g", geodb),
            ReportItem("SM1", self.SM1, "Site-adjusted MCEr, 1 s", "g", geodb),
            ReportItem("SDS", self.SDS, "(2/3) SMS", "g",
                       ref(EDITION, "11.4.4", equation="11.4-1")),
            ReportItem("SD1", self.SD1, "(2/3) SM1", "g",
                       ref(EDITION, "11.4.4", equation="11.4-2")),
            ReportItem("T0", cp.T0, "0.2 SD1/SDS", "s",
                       ref(EDITION, "11.4.5.2")),
            ReportItem("Ts", cp.Ts, "SD1/SDS", "s", ref(EDITION, "11.4.5.2")),
            ReportItem("TL", self.TL, "Long-period transition", "s",
                       ref(EDITION, "11.4.5.2", figure="22-14 to 22-17")),
            ReportItem("SDC", self.seismic_design_category.value,
                       "Seismic design category", "",
                       ref(EDITION, "11.6", table="11.6-1/11.6-2")),
        ]
        notes: list[str] = []
        if self.basis is SpectrumBasis.TWO_PERIOD:
            notes.append(
                "Two-period basis: §11.4.5 Exception 2 permits this only where "
                "multi-period values are unavailable from the USGS Seismic "
                "Design Geodatabase. Confirm that is the case."
            )
        if self.S1 is None:
            notes.append(
                "S1 was not supplied, so the §11.6 'S1 >= 0.75 implies SDC E/F' "
                "override could not be evaluated. The SDC shown reflects only "
                "Tables 11.6-1 and 11.6-2."
            )
        if self.default_site_conditions:
            notes.append(
                "Default site conditions (§11.4.2.1): MCEr accelerations must be "
                "the most critical of Site Classes C, CD and D at each period. "
                "Verify the supplied ordinates already envelope those three."
            )
        if self.sc == "F":
            notes.append(
                "Site Class F: a site response analysis is required per §21.1 "
                "unless exempted by §20.3.1 (§11.4.7)."
            )
        return Report(
            title=f"ASCE/SEI 7-22 seismic ground motion — Site Class {self.sc}",
            items=tuple(items),
            notes=tuple(notes),
        )

    def __str__(self) -> str:
        return self.report().to_text()


def _infer_parameters(
    T: np.ndarray, Sa_mcer: np.ndarray, site_class: SiteClass | str
) -> tuple[float, float]:
    """Infer ``SMS``/``SM1`` from MCEr ordinates using the §21.4 rules.

    §21.4 defines ``SDS`` as 90% of the maximum ``Sa`` over 0.2-5 s, and
    ``SD1`` as 90% of the maximum ``T*Sa`` over 1-2 s (or 1-5 s for softer
    sites), but not less than ``Sa`` at 1 s. ``SMS``/``SM1`` are 1.5x those.
    Applied here to the MCEr ordinates directly, which is equivalent.

    Only a fallback: §11.4.3 expects these read from the geodatabase.
    """
    sc = SiteClass(site_class).value
    # §21.4 widens the SD1 window for sites softer than roughly vs30 = 1,450 ft/s,
    # which in 7-22 terms is Site Class C and softer.
    upper = 2.0 if sc in ("A", "B", "BC") else 5.0

    short = (T >= 0.2) & (T <= 5.0)
    SMS = 0.9 * float(np.max(Sa_mcer[short])) if np.any(short) else float(np.max(Sa_mcer))

    window = (T >= 1.0) & (upper >= T)
    if np.any(window):
        SM1 = 0.9 * float(np.max(T[window] * Sa_mcer[window]))
        at_1s = float(np.interp(1.0, T, Sa_mcer))
        SM1 = max(SM1, at_1s)
    else:
        SM1 = float(np.interp(1.0, T, Sa_mcer))
    return SMS, SM1
