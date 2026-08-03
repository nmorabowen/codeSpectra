"""Matplotlib rendering for spectra. Requires ``pip install 'codeSpectra[plot]'``.

Kept deliberately thin: it draws curves and, optionally, annotates the control
periods that define each branch. Anything fancier belongs in the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core.spectrum import Spectrum
from .core.units import AccelUnit, from_g

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from matplotlib.axes import Axes

__all__ = ["compare", "plot_spectrum"]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib: pip install 'codeSpectra[plot]'"
        ) from exc
    return plt


def plot_spectrum(
    spectrum: Spectrum,
    ax: Axes | None = None,
    *,
    n: int = 600,
    t_max: float | None = None,
    unit: AccelUnit | str = AccelUnit.G,
    label: str | None = None,
    show_control_periods: bool = False,
    **kwargs: Any,
) -> Axes:
    """Draw ``spectrum`` on ``ax``, creating a figure if none is supplied."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    T, Sa = spectrum.sample(n=n, t_max=t_max)
    unit_enum = AccelUnit(unit)
    kwargs.setdefault("linewidth", 1.6)
    ax.plot(T, from_g(Sa, unit_enum), label=label or spectrum.meta.title, **kwargs)

    if show_control_periods:
        for name, value in spectrum.control_periods.items():
            if value <= 0.0 or value > float(T[-1]):
                continue
            ax.axvline(value, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
            ax.annotate(
                f"{name} = {value:.3g} s",
                xy=(value, ax.get_ylim()[1]),
                xytext=(3, -12),
                textcoords="offset points",
                rotation=90,
                fontsize=8,
                color="0.35",
                va="top",
            )

    ax.set_xlabel("Period, T (s)")
    ax.set_ylabel(f"Spectral acceleration, Sa ({unit_enum.label})")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    return ax


def compare(
    spectra: Sequence[Spectrum],
    ax: Axes | None = None,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Axes:
    """Overlay several spectra — typically the same site under different codes."""
    if labels is not None and len(labels) != len(spectra):
        raise ValueError(
            f"Got {len(labels)} labels for {len(spectra)} spectra."
        )
    for i, spectrum in enumerate(spectra):
        ax = plot_spectrum(
            spectrum,
            ax=ax,
            label=labels[i] if labels is not None else None,
            **kwargs,
        )
    assert ax is not None
    if title:
        ax.set_title(title)
    return ax
