"""ASCE/SEI 7-10 seismic ground motion (Chapter 11).

Retained for existing-building work and ASCE 41 tie-ins. The spectrum shape is
identical to 7-16; what differs is the site coefficient tables (five columns
rather than six, and Site Class E fully tabulated), the absence of the
default-Site-Class-D floor on ``Fa``, and the clause numbering — 7-10 puts the
site-specific trigger at §11.4.7 and has no §11.9 vertical spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ...core.control import ControlPeriods
from ...core.exceptions import InvalidInput
from ...core.references import ClauseRef
from ...core.reports import Report, ReportItem
from ...core.spectrum import AnalyticSpectrum, SpectrumKind
from ...core.tables import InterpolatedTable
from ._shared import (
    STANDARD,
    RiskCategory,
    SeismicDesignCategory,
    SiteClass,
    ref,
    seismic_design_category,
    two_period_spectrum,
)

__all__ = ["ASCE7_10", "FA_TABLE", "FV_TABLE"]

EDITION = "7-10"

_PERMITTED_SITE_CLASSES = ("A", "B", "C", "D", "E", "F")
_REMEDY = "Perform the site-specific ground motion procedures of §11.4.7 / Chapter 21."

#: Table 11.4-1 — Site coefficient Fa (ASCE 7-10, p. 54).
FA_TABLE = InterpolatedTable(
    name="Fa",
    row_label="site class",
    col_label="Ss",
    columns=(0.25, 0.5, 0.75, 1.0, 1.25),
    rows={
        "A": (0.8, 0.8, 0.8, 0.8, 0.8),
        "B": (1.0, 1.0, 1.0, 1.0, 1.0),
        "C": (1.2, 1.2, 1.1, 1.0, 1.0),
        "D": (1.6, 1.4, 1.2, 1.1, 1.0),
        "E": (2.5, 1.7, 1.2, 0.9, 0.9),
        "F": (None, None, None, None, None),
    },
    ref=ref(EDITION, "11.4.3", "Site coefficient Fa", table="11.4-1"),
    site_specific_remedy=_REMEDY,
)

#: Table 11.4-2 — Site coefficient Fv (ASCE 7-10, p. 54).
FV_TABLE = InterpolatedTable(
    name="Fv",
    row_label="site class",
    col_label="S1",
    columns=(0.1, 0.2, 0.3, 0.4, 0.5),
    rows={
        "A": (0.8, 0.8, 0.8, 0.8, 0.8),
        "B": (1.0, 1.0, 1.0, 1.0, 1.0),
        "C": (1.7, 1.6, 1.5, 1.4, 1.3),
        "D": (2.4, 2.0, 1.8, 1.6, 1.5),
        "E": (3.5, 3.2, 2.8, 2.4, 2.4),
        "F": (None, None, None, None, None),
    },
    ref=ref(EDITION, "11.4.3", "Site coefficient Fv", table="11.4-2"),
    site_specific_remedy=_REMEDY,
)


@dataclass(frozen=True)
class ASCE7_10:
    """Seismic ground motion parameters and spectra per ASCE/SEI 7-10.

    Examples
    --------
    >>> site = ASCE7_10(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
    >>> site.Fa, site.Fv
    (1.0, 1.5)
    >>> round(site.SDS, 4), round(site.SD1, 4)
    (1.0, 0.6)
    """

    Ss: float
    S1: float
    site_class: SiteClass | str = SiteClass.D
    TL: float = 8.0
    risk_category: RiskCategory | str = RiskCategory.II
    Fa_override: float | None = None
    Fv_override: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_class", SiteClass(self.site_class))
        object.__setattr__(self, "risk_category", RiskCategory(self.risk_category))
        if self.sc not in _PERMITTED_SITE_CLASSES:
            raise InvalidInput(
                f"ASCE 7-10 defines Site Classes {_PERMITTED_SITE_CLASSES}; "
                f"got {self.sc!r}."
            )
        if self.Ss < 0.0 or self.S1 < 0.0:
            raise InvalidInput("Ss and S1 must be non-negative.")
        if self.TL <= 0.0:
            raise InvalidInput("TL must be positive.")

    @property
    def sc(self) -> str:
        """Site class as a plain string."""
        return SiteClass(self.site_class).value

    @cached_property
    def Fa(self) -> float:
        """Short-period site coefficient (Table 11.4-1)."""
        if self.Fa_override is not None:
            return self.Fa_override
        return FA_TABLE.lookup(self.sc, self.Ss)

    @cached_property
    def Fv(self) -> float:
        """Long-period site coefficient (Table 11.4-2)."""
        if self.Fv_override is not None:
            return self.Fv_override
        return FV_TABLE.lookup(self.sc, self.S1)

    @cached_property
    def SMS(self) -> float:
        """``Fa * Ss`` (Eq. 11.4-1)."""
        return self.Fa * self.Ss

    @cached_property
    def SM1(self) -> float:
        """``Fv * S1`` (Eq. 11.4-2)."""
        return self.Fv * self.S1

    @cached_property
    def SDS(self) -> float:
        """``(2/3) SMS`` (Eq. 11.4-3)."""
        return (2.0 / 3.0) * self.SMS

    @cached_property
    def SD1(self) -> float:
        """``(2/3) SM1`` (Eq. 11.4-4)."""
        return (2.0 / 3.0) * self.SM1

    @cached_property
    def control_periods(self) -> ControlPeriods:
        """``T0``, ``Ts`` and ``TL`` (§11.4.5)."""
        Ts = self.SD1 / self.SDS if self.SDS > 0.0 else 0.0
        return ControlPeriods(T0=0.2 * Ts, Ts=Ts, TL=self.TL)

    @property
    def Ie(self) -> float:
        """Seismic importance factor (Table 1.5-2)."""
        return RiskCategory(self.risk_category).importance_factor

    @cached_property
    def seismic_design_category(self) -> SeismicDesignCategory:
        """Seismic design category per §11.6."""
        return seismic_design_category(
            self.SDS, self.SD1, self.S1, RiskCategory(self.risk_category)
        )

    def design_spectrum(self, *, t_max: float | None = None) -> AnalyticSpectrum:
        """The §11.4.5 design response spectrum (Fig. 11.4-1)."""
        return two_period_spectrum(
            edition=EDITION,
            SDS=self.SDS,
            SD1=self.SD1,
            TL=self.TL,
            kind=SpectrumKind.DESIGN,
            label=f"ASCE 7-10 design spectrum (Site Class {self.sc})",
            parameters=self._parameters(),
            refs=(ref(EDITION, "11.4.5", "Design response spectrum", figure="11.4-1"),),
            t_max=t_max,
        )

    def mcer_spectrum(self, *, t_max: float | None = None) -> AnalyticSpectrum:
        """The MCEr response spectrum: 1.5x the design spectrum (§11.4.6)."""
        return two_period_spectrum(
            edition=EDITION,
            SDS=self.SMS,
            SD1=self.SM1,
            TL=self.TL,
            kind=SpectrumKind.MCER,
            label=f"ASCE 7-10 MCEr spectrum (Site Class {self.sc})",
            parameters=self._parameters(),
            refs=(ref(EDITION, "11.4.6", "MCEr response spectrum"),),
            t_max=t_max,
        )

    def _parameters(self) -> dict[str, float | str]:
        return {
            "Ss": self.Ss, "S1": self.S1, "site_class": self.sc,
            "Fa": self.Fa, "Fv": self.Fv, "SMS": self.SMS, "SM1": self.SM1,
            "SDS": self.SDS, "SD1": self.SD1, "TL": self.TL,
        }

    def report(self) -> Report:
        """A citation-carrying record of the parameter derivation."""
        cp = self.control_periods
        items = [
            ReportItem("Ss", self.Ss, "Mapped MCEr short-period acceleration", "g",
                       ref(EDITION, "11.4.1", "Mapped acceleration parameters")),
            ReportItem("S1", self.S1, "Mapped MCEr 1-s acceleration", "g",
                       ref(EDITION, "11.4.1", "Mapped acceleration parameters")),
            ReportItem("Site Class", self.sc, "Chapter 20 site class", "",
                       ref(EDITION, "11.4.2", "Site class")),
            ReportItem("Ie", self.Ie, "Seismic importance factor", "",
                       ClauseRef(STANDARD, EDITION, "11.5.1", table="1.5-2")),
            ReportItem("Fa", self.Fa, "Short-period site coefficient", "",
                       FA_TABLE.ref),
            ReportItem("Fv", self.Fv, "Long-period site coefficient", "",
                       FV_TABLE.ref),
            ReportItem("SMS", self.SMS, "Fa x Ss", "g",
                       ref(EDITION, "11.4.3", equation="11.4-1")),
            ReportItem("SM1", self.SM1, "Fv x S1", "g",
                       ref(EDITION, "11.4.3", equation="11.4-2")),
            ReportItem("SDS", self.SDS, "(2/3) SMS", "g",
                       ref(EDITION, "11.4.4", equation="11.4-3")),
            ReportItem("SD1", self.SD1, "(2/3) SM1", "g",
                       ref(EDITION, "11.4.4", equation="11.4-4")),
            ReportItem("T0", cp.T0, "0.2 SD1/SDS", "s",
                       ref(EDITION, "11.4.5", "Design response spectrum")),
            ReportItem("Ts", cp.Ts, "SD1/SDS", "s",
                       ref(EDITION, "11.4.5", "Design response spectrum")),
            ReportItem("TL", self.TL, "Long-period transition", "s",
                       ref(EDITION, "11.4.5", figure="22-12 to 22-16")),
            ReportItem("SDC", self.seismic_design_category.value,
                       "Seismic design category", "",
                       ref(EDITION, "11.6", table="11.6-1/11.6-2")),
        ]
        notes = []
        if self.sc == "F":
            notes.append(
                "Site Class F: site-specific ground motion procedures are "
                "required per §11.4.7 / Chapter 21."
            )
        return Report(
            title=f"ASCE/SEI 7-10 seismic ground motion — Site Class {self.sc}",
            items=tuple(items),
            notes=tuple(notes),
        )

    def __str__(self) -> str:
        return self.report().to_text()
