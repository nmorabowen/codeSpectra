"""Write spectra in the formats analysis packages ingest.

Every writer samples the spectrum on a grid that includes its control periods,
so the exported curve reproduces the corners of the code figure rather than
chamfering them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .core.spectrum import Spectrum
from .core.units import AccelUnit, from_g

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "to_csv",
    "to_etabs",
    "to_json",
    "to_opensees",
    "to_sap2000",
]


#: Decimal places every writer emits. Periods are deduplicated at this
#: precision so a file never carries two rows with the same abscissa.
PRECISION = 6


def _sampled(
    spectrum: Spectrum,
    periods: ArrayLike | None,
    n: int,
    t_max: float | None,
    unit: AccelUnit | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the spectrum, then collapse periods equal at output precision.

    ``Spectrum.grid`` straddles each control period by a hair so a slope
    discontinuity plots as a corner. Those straddle points round to the same
    value at :data:`PRECISION`, which would put duplicate abscissae in the
    exported file — invalid for an OpenSees ``Path`` series and meaningless
    for ETABS. Since a code spectrum is continuous, dropping them loses
    nothing.
    """
    if periods is None:
        T = spectrum.grid(n=n, t_max=t_max)
    else:
        T = np.atleast_1d(np.asarray(periods, dtype=float))
    Sa = from_g(np.atleast_1d(spectrum.at(T)), unit)

    rounded = np.round(T, PRECISION)
    _, first = np.unique(rounded, return_index=True)
    keep = np.sort(first)
    return rounded[keep], Sa[keep]


def to_csv(
    spectrum: Spectrum,
    path: str | Path,
    *,
    periods: ArrayLike | None = None,
    n: int = 200,
    t_max: float | None = None,
    unit: AccelUnit | str = AccelUnit.G,
) -> Path:
    """Write ``T,Sa`` as CSV with a header naming the unit."""
    T, Sa = _sampled(spectrum, periods, n, t_max, unit)
    out = Path(path)
    lines = [f"T [s],Sa [{AccelUnit(unit).label}]"]
    lines += [f"{t:.6f},{s:.6f}" for t, s in zip(T, Sa, strict=True)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def to_json(
    spectrum: Spectrum,
    path: str | Path,
    *,
    periods: ArrayLike | None = None,
    n: int = 200,
    t_max: float | None = None,
    unit: AccelUnit | str = AccelUnit.G,
) -> Path:
    """Write the ordinates plus the full provenance metadata as JSON."""
    import json

    T, Sa = _sampled(spectrum, periods, n, t_max, unit)
    meta = spectrum.meta
    payload = {
        "standard": meta.standard,
        "edition": meta.edition,
        "kind": meta.kind.value,
        "label": meta.title,
        "damping": meta.damping,
        "unit": AccelUnit(unit).value,
        "control_periods": spectrum.control_periods.as_dict(),
        "parameters": meta.parameters,
        "references": [str(r) for r in meta.refs],
        "T": [round(float(t), 6) for t in T],
        "Sa": [round(float(s), 6) for s in Sa],
    }
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def to_etabs(
    spectrum: Spectrum,
    path: str | Path,
    *,
    periods: ArrayLike | None = None,
    n: int = 200,
    t_max: float | None = None,
    unit: AccelUnit | str = AccelUnit.G,
) -> Path:
    """Write an ETABS response-spectrum function file.

    ETABS reads a plain two-column ``period value`` text file; comment lines
    beginning with ``#`` carry the provenance. Import via *Define > Functions
    > Response Spectrum > From File*, and set the function's units to match.
    """
    T, Sa = _sampled(spectrum, periods, n, t_max, unit)
    header = [
        f"# {spectrum.meta.title}",
        f"# Units: period [s], acceleration [{AccelUnit(unit).label}]",
        f"# Damping: {spectrum.meta.damping:.0%}",
    ]
    header += [f"# {r}" for r in spectrum.meta.refs]
    body = [f"{t:.6f}\t{s:.6f}" for t, s in zip(T, Sa, strict=True)]
    out = Path(path)
    out.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return out


def to_sap2000(
    spectrum: Spectrum,
    path: str | Path,
    *,
    periods: ArrayLike | None = None,
    n: int = 200,
    t_max: float | None = None,
    unit: AccelUnit | str = AccelUnit.G,
) -> Path:
    """Write a SAP2000 response-spectrum function file.

    Same two-column layout as ETABS; kept separate because the two products
    differ in how they treat the header and in their default units.
    """
    return to_etabs(spectrum, path, periods=periods, n=n, t_max=t_max, unit=unit)


def to_opensees(
    spectrum: Spectrum,
    path: str | Path,
    *,
    periods: ArrayLike | None = None,
    n: int = 200,
    t_max: float | None = None,
    unit: AccelUnit | str = AccelUnit.G,
    series_tag: int = 1,
    style: str = "tcl",
) -> Path:
    """Write the spectrum as an OpenSees ``Path`` time series.

    Parameters
    ----------
    style
        ``"tcl"`` emits a ``timeSeries Path`` command; ``"python"`` emits the
        equivalent ``ops.timeSeries(...)`` call for OpenSeesPy.

    Notes
    -----
    A ``Path`` series indexed by period is the usual way to hand a design
    spectrum to a response-spectrum analysis in OpenSees. It is *not* a ground
    motion record — do not feed it to a ``UniformExcitation``.
    """
    T, Sa = _sampled(spectrum, periods, n, t_max, unit)
    t_list = " ".join(f"{t:.6f}" for t in T)
    s_list = " ".join(f"{s:.6f}" for s in Sa)
    comment = "#"
    header = [
        f"{comment} {spectrum.meta.title}",
        f"{comment} period [s] vs acceleration [{AccelUnit(unit).label}]",
    ]
    if style == "tcl":
        body = [
            f"timeSeries Path {series_tag} -time {{{t_list}}} -values {{{s_list}}}"
        ]
    elif style == "python":
        body = [
            "import openseespy.opensees as ops",
            f"ops.timeSeries('Path', {series_tag},",
            f"                '-time', {list(np.round(T, 6))},",
            f"                '-values', {list(np.round(Sa, 6))})",
        ]
    else:
        raise ValueError(f"style must be 'tcl' or 'python', got {style!r}.")
    out = Path(path)
    out.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return out
