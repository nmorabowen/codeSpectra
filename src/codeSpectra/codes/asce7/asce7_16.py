"""ASCE/SEI 7-16 seismic ground motion (Chapter 11).

Input contract: the user supplies the mapped parameters ``Ss`` and ``S1`` and
the long-period transition ``TL`` (Figs. 22-14 to 22-17), plus the site class.
The site coefficients ``Fa`` and ``Fv`` come from Tables 11.4-1 and 11.4-2.

Where the tables defer to §11.4.8, this module raises
:class:`~codeSpectra.core.exceptions.SiteSpecificRequired` rather than
inventing a coefficient. Two escape hatches exist, both explicit:
``allow_site_specific_exception`` applies the §11.4.8 Exception 1 substitution
(``Fa`` of Site Class C), and ``Fa``/``Fv`` may be supplied directly by an
engineer who has performed the study.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from ...core.control import ControlPeriods
from ...core.exceptions import InvalidInput, SiteSpecificRequired
from ...core.references import ClauseRef
from ...core.reports import Report, ReportItem
from ...core.spectrum import AnalyticSpectrum, Spectrum, SpectrumKind, SpectrumMeta
from . import _tables_7_16 as T716
from ._shared import (
    STANDARD,
    RiskCategory,
    SeismicDesignCategory,
    SiteClass,
    ref,
    seismic_design_category,
    two_period_spectrum,
)

__all__ = ["ASCE7_16"]

EDITION = "7-16"

_PERMITTED_SITE_CLASSES = ("A", "B", "C", "D", "E", "F")

_R = {
    "site_class": ref(EDITION, "11.4.3", "Site class", table="20.3-1"),
    "coeffs": ref(EDITION, "11.4.4", "Site coefficients", table="11.4-1/11.4-2"),
    "SMS": ref(EDITION, "11.4.4", "MCEr short-period parameter", equation="11.4-1"),
    "SM1": ref(EDITION, "11.4.4", "MCEr 1-s parameter", equation="11.4-2"),
    "SDS": ref(EDITION, "11.4.5", "Design short-period parameter", equation="11.4-3"),
    "SD1": ref(EDITION, "11.4.5", "Design 1-s parameter", equation="11.4-4"),
    "spectrum": ref(EDITION, "11.4.6", "Design response spectrum", figure="11.4-1"),
    "mcer": ref(EDITION, "11.4.7", "MCEr response spectrum"),
    "gmha": ref(EDITION, "11.4.8", "Site-specific ground motion procedures"),
    "sdc": ref(EDITION, "11.6", "Seismic design category", table="11.6-1/11.6-2"),
    "vertical": ref(EDITION, "11.9.2", "MCEr vertical response spectrum"),
    "vertical_design": ref(EDITION, "11.9.3", "Design vertical response spectrum"),
    "default_D": ref(EDITION, "11.4.4", "Fa >= 1.2 where Site Class D is the default"),
    "rock_B": ref(EDITION, "11.4.3", "Fa = Fv = 1.0 for unmeasured Site Class B rock"),
}


@dataclass(frozen=True, slots=True)
class _Coefficients:
    """Site coefficients plus any narrative the modifiers generated."""

    Fa: float
    Fv: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ASCE7_16:
    """Seismic ground motion parameters and spectra per ASCE/SEI 7-16.

    Parameters
    ----------
    Ss, S1
        Mapped risk-targeted MCEr spectral response acceleration parameters at
        short period and 1 s, in g (§11.4.2).
    site_class
        Site class per Chapter 20. One of A-F.
    TL
        Long-period transition period, s (Figs. 22-14 to 22-17). Defaults to
        8 s, a common value for much of the western United States; supply the
        mapped value for real work.
    risk_category
        Risk category per Table 1.5-1, used for the seismic design category
        and the importance factor.
    default_site_class
        True if Site Class D was assumed because soil properties are unknown
        (§11.4.3). Triggers the ``Fa >= 1.2`` floor.
    measured_shear_wave_velocity
        False if Site Class B rock was identified without velocity
        measurements, which sets ``Fa = Fv = 1.0`` (§11.4.3).
    seismically_isolated
        True for seismically isolated structures or structures with damping
        systems, which changes the §11.4.8 triggers.
    allow_site_specific_exception
        Apply the §11.4.8 Exception 1 substitution (use the Site Class C value
        of ``Fa`` for Site Class E with ``Ss >= 1.0``) instead of raising.
    Fa_override, Fv_override
        Explicit site coefficient overrides, for an engineer who has performed
        the required study. Bypass the tables entirely.

    Examples
    --------
    >>> site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
    >>> round(site.SDS, 3), round(site.SD1, 3)
    (1.0, 0.68)
    >>> round(site.design_spectrum().at(1.0), 4)
    0.68
    """

    Ss: float
    S1: float
    site_class: SiteClass | str = SiteClass.D
    TL: float = 8.0
    risk_category: RiskCategory | str = RiskCategory.II
    default_site_class: bool = False
    measured_shear_wave_velocity: bool = True
    seismically_isolated: bool = False
    allow_site_specific_exception: bool = False
    Fa_override: float | None = None
    Fv_override: float | None = None

    # -- Validation -------------------------------------------------------
    def __post_init__(self) -> None:
        object.__setattr__(self, "site_class", SiteClass(self.site_class))
        object.__setattr__(self, "risk_category", RiskCategory(self.risk_category))
        if self.sc not in _PERMITTED_SITE_CLASSES:
            raise InvalidInput(
                f"ASCE 7-16 defines Site Classes {_PERMITTED_SITE_CLASSES}; "
                f"got {self.sc!r}. Site classes BC, CD and DE were "
                "introduced in ASCE 7-22 — use the ASCE7_22 class for those."
            )
        if self.Ss < 0.0 or self.S1 < 0.0:
            raise InvalidInput("Ss and S1 must be non-negative.")
        if self.Ss < self.S1:
            raise InvalidInput(
                f"S1 ({self.S1:g}) exceeds Ss ({self.Ss:g}); check the mapped values."
            )
        if self.TL <= 0.0:
            raise InvalidInput("TL must be positive.")

    @property
    def sc(self) -> str:
        """Site class as a plain string, e.g. ``"D"``."""
        return SiteClass(self.site_class).value

    # -- Site coefficients -------------------------------------------------
    @cached_property
    def site_coefficients(self) -> _Coefficients:
        """``Fa`` and ``Fv`` including every §11.4.3/§11.4.4 modifier."""
        notes: list[str] = []
        sc = self.sc

        if self.default_site_class and sc != "D":
            raise InvalidInput(
                "default_site_class=True only applies to Site Class D (§11.4.3)."
            )

        unmeasured_rock = sc == "B" and not self.measured_shear_wave_velocity
        if unmeasured_rock:
            Fa, Fv = 1.0, 1.0
            notes.append(
                "Site Class B rock without measured shear wave velocity: "
                "Fa = Fv = 1.0 per §11.4.3."
            )
        else:
            Fa = self.Fa_override if self.Fa_override is not None else self._lookup_Fa()
            Fv = (
                self.Fv_override
                if self.Fv_override is not None
                else T716.FV_TABLE.lookup(sc, self.S1)
            )

            if self.Fa_override is None and self.default_site_class and Fa < 1.2:
                notes.append(
                    f"Fa raised from {Fa:g} to 1.2: Site Class D is the default "
                    "site class (§11.4.4)."
                )
                Fa = 1.2

        return _Coefficients(Fa=Fa, Fv=Fv, notes=tuple(notes))

    def _lookup_Fa(self) -> float:
        sc = self.sc
        try:
            return T716.FA_TABLE.lookup(sc, self.Ss)
        except SiteSpecificRequired:
            # §11.4.8 Exception 1: Site Class E with Ss >= 1.0 needs no ground
            # motion hazard analysis provided Fa is taken as that of Site Class C.
            if self.allow_site_specific_exception and sc == "E" and self.Ss >= 1.0:
                return T716.FA_TABLE.lookup("C", self.Ss)
            raise

    @property
    def Fa(self) -> float:
        """Short-period site coefficient actually used (Table 11.4-1)."""
        return self.site_coefficients.Fa

    @property
    def Fv(self) -> float:
        """Long-period site coefficient actually used (Table 11.4-2)."""
        return self.site_coefficients.Fv

    # -- Acceleration parameters -------------------------------------------
    @cached_property
    def SMS(self) -> float:
        """Site-adjusted MCEr short-period acceleration, ``Fa*Ss`` (Eq. 11.4-1)."""
        return self.Fa * self.Ss

    @cached_property
    def SM1(self) -> float:
        """Site-adjusted MCEr 1-s acceleration, ``Fv*S1`` (Eq. 11.4-2)."""
        return self.Fv * self.S1

    @cached_property
    def SDS(self) -> float:
        """Design short-period acceleration, ``(2/3) SMS`` (Eq. 11.4-3)."""
        return (2.0 / 3.0) * self.SMS

    @cached_property
    def SD1(self) -> float:
        """Design 1-s acceleration, ``(2/3) SM1`` (Eq. 11.4-4)."""
        return (2.0 / 3.0) * self.SM1

    @cached_property
    def control_periods(self) -> ControlPeriods:
        """``T0``, ``Ts`` and ``TL`` (§11.4.6)."""
        Ts = self.SD1 / self.SDS if self.SDS > 0.0 else 0.0
        return ControlPeriods(T0=0.2 * Ts, Ts=Ts, TL=self.TL)

    @property
    def Ie(self) -> float:
        """Seismic importance factor (Table 1.5-2)."""
        assert isinstance(self.risk_category, RiskCategory)
        return self.risk_category.importance_factor

    @cached_property
    def seismic_design_category(self) -> SeismicDesignCategory:
        """Seismic design category per §11.6."""
        assert isinstance(self.risk_category, RiskCategory)
        return seismic_design_category(self.SDS, self.SD1, self.S1, self.risk_category)

    # -- Spectra ------------------------------------------------------------
    def design_spectrum(self, *, t_max: float | None = None) -> AnalyticSpectrum:
        """The §11.4.6 design response spectrum (Fig. 11.4-1)."""
        return two_period_spectrum(
            edition=EDITION,
            SDS=self.SDS,
            SD1=self.SD1,
            TL=self.TL,
            kind=SpectrumKind.DESIGN,
            label=f"ASCE 7-16 design spectrum (Site Class {self.sc})",
            parameters=self._parameters(),
            refs=(_R["spectrum"], _R["SDS"], _R["SD1"], _R["coeffs"]),
            t_max=t_max,
        )

    def mcer_spectrum(self, *, t_max: float | None = None) -> Spectrum:
        """The MCEr response spectrum: 1.5x the design spectrum (§11.4.7)."""
        base = two_period_spectrum(
            edition=EDITION,
            SDS=self.SMS,
            SD1=self.SM1,
            TL=self.TL,
            kind=SpectrumKind.MCER,
            label=f"ASCE 7-16 MCEr spectrum (Site Class {self.sc})",
            parameters=self._parameters(),
            refs=(_R["mcer"], _R["SMS"], _R["SM1"]),
            t_max=t_max,
        )
        return base

    def vertical_spectrum(self, *, design: bool = True) -> AnalyticSpectrum:
        """The §11.9 vertical response spectrum, valid to ``Tv = 2.0`` s.

        Parameters
        ----------
        design
            True returns the §11.9.3 design vertical spectrum (2/3 of MCEr);
            False returns the §11.9.2 MCEr vertical spectrum.

        Notes
        -----
        §11.9.1 restricts this procedure to Seismic Design Categories C
        through F, and §11.9.2 requires a site-specific procedure beyond
        ``Tv = 2.0`` s, so the spectrum is reported only to 2 s.
        """
        Cv = T716.CV_TABLE.lookup(T716.cv_row_for(self.sc), self.Ss)
        SMS = self.SMS
        factor = (2.0 / 3.0) if design else 1.0

        def evaluate(Tv: NDArray[np.float64]) -> NDArray[np.float64]:
            Tv = np.asarray(Tv, dtype=float)
            out = np.full(Tv.shape, 0.3 * Cv * SMS, dtype=float)
            band2 = (Tv > 0.025) & (Tv <= 0.05)
            out[band2] = 20.0 * Cv * SMS * (Tv[band2] - 0.025) + 0.3 * Cv * SMS
            band3 = (Tv > 0.05) & (Tv <= 0.15)
            out[band3] = 0.8 * Cv * SMS
            band4 = Tv > 0.15
            safe = np.where(Tv > 0.0, Tv, 1.0)
            out[band4] = 0.8 * Cv * SMS * (0.15 / safe[band4]) ** 0.75
            return out * factor

        kind = SpectrumKind.VERTICAL
        meta = SpectrumMeta(
            standard=STANDARD,
            edition=EDITION,
            kind=kind,
            label=f"ASCE 7-16 {'design' if design else 'MCEr'} vertical spectrum",
            control_periods=ControlPeriods(Tv1=0.025, Tv2=0.05, Tv3=0.15, Tv_max=2.0),
            parameters={**self._parameters(), "Cv": Cv},
            refs=(_R["vertical_design"] if design else _R["vertical"],),
        )
        return AnalyticSpectrum(evaluate, meta=meta, t_max=2.0)

    # -- Site-specific requirements ----------------------------------------
    @cached_property
    def site_specific_triggers(self) -> tuple[tuple[str, ClauseRef], ...]:
        """Every §11.4.8 condition this site meets, with its citation.

        Returned rather than raised, because several triggers have exceptions
        that let an engineer proceed with the general procedure. Consult these
        before relying on the design spectrum.
        """
        out: list[tuple[str, ClauseRef]] = []
        sc = self.sc
        if sc == "F":
            out.append((
                "Site Class F: a site response analysis is required per §21.1 "
                "unless exempted by §20.3.1.",
                ref(EDITION, "11.4.8", "Site Class F"),
            ))
        if self.seismically_isolated and self.S1 >= 0.6:
            out.append((
                "Seismically isolated or damped structure with S1 >= 0.6: a "
                "ground motion hazard analysis is required per §21.2.",
                _R["gmha"],
            ))
        if sc == "E" and self.Ss >= 1.0:
            out.append((
                "Site Class E with Ss >= 1.0: ground motion hazard analysis "
                "required, unless Fa is taken as that of Site Class C "
                "(Exception 1).",
                _R["gmha"],
            ))
        if sc in ("D", "E") and self.S1 >= 0.2:
            out.append((
                f"Site Class {sc} with S1 >= 0.2: ground motion hazard analysis "
                "required, unless the Cs limits of Exception 2 (Site Class D) or "
                "the T <= Ts / ELF condition of Exception 3 (Site Class E) apply.",
                _R["gmha"],
            ))
        return tuple(out)

    # -- Reporting -----------------------------------------------------------
    def _parameters(self) -> dict[str, float | str]:
        return {
            "Ss": self.Ss,
            "S1": self.S1,
            "site_class": self.sc,
            "Fa": self.Fa,
            "Fv": self.Fv,
            "SMS": self.SMS,
            "SM1": self.SM1,
            "SDS": self.SDS,
            "SD1": self.SD1,
            "TL": self.TL,
        }

    def report(self) -> Report:
        """A citation-carrying record of the full parameter derivation."""
        cp = self.control_periods
        assert isinstance(self.risk_category, RiskCategory)
        items = [
            ReportItem("Ss", self.Ss, "Mapped MCEr short-period acceleration", "g",
                       ref(EDITION, "11.4.2", "Mapped acceleration parameters")),
            ReportItem("S1", self.S1, "Mapped MCEr 1-s acceleration", "g",
                       ref(EDITION, "11.4.2", "Mapped acceleration parameters")),
            ReportItem("Site Class", self.sc, "Chapter 20 site class",
                       "", _R["site_class"]),
            ReportItem("Risk Category", self.risk_category.value, "Table 1.5-1"),
            ReportItem("Ie", self.Ie, "Seismic importance factor", "",
                       ClauseRef(STANDARD, EDITION, "11.5.1", table="1.5-2")),
            ReportItem("Fa", self.Fa, "Short-period site coefficient", "",
                       T716.FA_TABLE.ref),
            ReportItem("Fv", self.Fv, "Long-period site coefficient", "",
                       T716.FV_TABLE.ref),
            ReportItem("SMS", self.SMS, "Fa x Ss", "g", _R["SMS"]),
            ReportItem("SM1", self.SM1, "Fv x S1", "g", _R["SM1"]),
            ReportItem("SDS", self.SDS, "(2/3) SMS", "g", _R["SDS"]),
            ReportItem("SD1", self.SD1, "(2/3) SM1", "g", _R["SD1"]),
            ReportItem("T0", cp.T0, "0.2 SD1/SDS", "s", _R["spectrum"]),
            ReportItem("Ts", cp.Ts, "SD1/SDS", "s", _R["spectrum"]),
            ReportItem("TL", self.TL, "Long-period transition", "s",
                       ref(EDITION, "11.4.6", "Long-period transition period",
                           figure="22-14 to 22-17")),
            ReportItem("SDC", self.seismic_design_category.value,
                       "Seismic design category", "", _R["sdc"]),
        ]
        notes = list(self.site_coefficients.notes)
        notes.extend(text for text, _ in self.site_specific_triggers)
        return Report(
            title=f"ASCE/SEI 7-16 seismic ground motion — Site Class "
                  f"{self.sc}",
            items=tuple(items),
            notes=tuple(notes),
        )

    def __str__(self) -> str:
        return self.report().to_text()
