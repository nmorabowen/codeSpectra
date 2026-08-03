"""Code-agnostic building blocks.

Nothing in this package knows that any particular design standard exists.
"""

from .control import ControlPeriods
from .exceptions import (
    CodeSpectraError,
    InvalidInput,
    SiteSpecificRequired,
    TableLookupError,
)
from .references import ClauseRef
from .reports import Report, ReportItem
from .spectrum import (
    AnalyticSpectrum,
    Spectrum,
    SpectrumKind,
    SpectrumMeta,
    TabulatedSpectrum,
)
from .tables import InterpolatedTable
from .units import STANDARD_GRAVITY, AccelUnit, from_g

__all__ = [
    "STANDARD_GRAVITY",
    "AccelUnit",
    "AnalyticSpectrum",
    "ClauseRef",
    "CodeSpectraError",
    "ControlPeriods",
    "InterpolatedTable",
    "InvalidInput",
    "Report",
    "ReportItem",
    "SiteSpecificRequired",
    "Spectrum",
    "SpectrumKind",
    "SpectrumMeta",
    "TableLookupError",
    "TabulatedSpectrum",
    "from_g",
]
