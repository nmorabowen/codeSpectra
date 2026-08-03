"""codeSpectra — codified seismic design response spectra.

Build a code-defined response spectrum from site parameters, then treat it as
a first-class object: scale it, reduce it by ``R``, envelope it, floor it
against a site-specific study, export it, plot it.

Quick start
-----------
>>> from codeSpectra import ASCE7_16
>>> site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
>>> round(site.SDS, 3)
1.0
>>> round(site.design_spectrum().at(1.0), 3)
0.68

Every derived quantity carries its clause::

    print(site.report().to_text())
"""

from .core import (
    AccelUnit,
    AnalyticSpectrum,
    ClauseRef,
    CodeSpectraError,
    ControlPeriods,
    InvalidInput,
    Report,
    SiteSpecificRequired,
    Spectrum,
    SpectrumKind,
    TabulatedSpectrum,
)

__version__ = "0.2.0"

__all__ = [
    "AccelUnit",
    "AnalyticSpectrum",
    "ClauseRef",
    "CodeSpectraError",
    "ControlPeriods",
    "InvalidInput",
    "Report",
    "SiteSpecificRequired",
    "Spectrum",
    "SpectrumKind",
    "TabulatedSpectrum",
    "__version__",
]


def __getattr__(name: str) -> object:
    """Lazily expose the code classes so importing the package stays cheap."""
    _codes = {
        "ASCE7_10": ("codes.asce7.asce7_10", "ASCE7_10"),
        "ASCE7_16": ("codes.asce7.asce7_16", "ASCE7_16"),
        "ASCE7_22": ("codes.asce7.asce7_22", "ASCE7_22"),
        "NECSEDS2015": ("codes.nec.nec_se_ds_2015", "NECSEDS2015"),
        "NCh433": ("codes.nch433.nch433", "NCh433"),
        "RiskCategory": ("codes.asce7._shared", "RiskCategory"),
        "SiteClass": ("codes.asce7._shared", "SiteClass"),
        "SeismicZone": ("codes.nec.nec_se_ds_2015", "SeismicZone"),
        "SoilType": ("codes.nec.nec_se_ds_2015", "SoilType"),
        "Region": ("codes.nec.nec_se_ds_2015", "Region"),
        "compare": ("plotting", "compare"),
    }
    if name in _codes:
        from importlib import import_module

        module_path, attr = _codes[name]
        return getattr(import_module(f".{module_path}", __package__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ += [
    "ASCE7_10",
    "ASCE7_16",
    "ASCE7_22",
    "NECSEDS2015",
    "NCh433",
    "Region",
    "RiskCategory",
    "SeismicZone",
    "SiteClass",
    "SoilType",
    "compare",
]
