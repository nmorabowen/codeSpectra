"""The central value object: a response spectrum.

Everything the code modules produce is a :class:`Spectrum`, and every
downstream operation — scaling, reduction by ``R``, enveloping, applying the
ASCE §21.3 80% floor, exporting, plotting — is a method on it. Two concrete
kinds share the interface:

``AnalyticSpectrum``
    Carries a closed-form evaluator, so ``at(T)`` is exact at any period and
    sampling is on demand. Used by every branch-defined code spectrum.

``TabulatedSpectrum``
    ``(T, Sa)`` ordinates with linear-in-T interpolation. Used by ASCE 7-22
    multi-period spectra, site-specific studies, and imported curves.

All operations are non-mutating: they return new spectra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .control import ControlPeriods
from .exceptions import InvalidInput
from .references import ClauseRef
from .units import AccelUnit, from_g

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes

__all__ = [
    "AnalyticSpectrum",
    "Spectrum",
    "SpectrumKind",
    "SpectrumMeta",
    "TabulatedSpectrum",
]


class SpectrumKind(str, Enum):
    """What hazard level / reduction state a spectrum represents."""

    ELASTIC = "elastic"
    """Elastic, at the design hazard level, unreduced (NEC Sa, NCh433 Sa)."""

    DESIGN = "design"
    """ASCE design spectrum: 2/3 of MCEr, unreduced by R."""

    MCER = "mcer"
    """Risk-targeted maximum considered earthquake spectrum."""

    INELASTIC = "inelastic"
    """Reduced by a response modification factor."""

    VERTICAL = "vertical"
    """Vertical component spectrum."""

    SITE_SPECIFIC = "site_specific"
    """Produced by a site response / ground motion hazard study."""

    DISPLACEMENT = "displacement"
    """Spectral displacement rather than acceleration."""


@dataclass(frozen=True, slots=True)
class SpectrumMeta:
    """Provenance travelling with a spectrum.

    ``parameters`` holds the scalar inputs that generated the curve (``SDS``,
    ``Z``, ``Fa``, ``R`` ...) so a report can be rendered without re-deriving
    anything.
    """

    standard: str = ""
    edition: str = ""
    kind: SpectrumKind = SpectrumKind.ELASTIC
    damping: float = 0.05
    label: str = ""
    control_periods: ControlPeriods = field(default_factory=ControlPeriods)
    parameters: dict[str, float | str] = field(default_factory=dict)
    refs: tuple[ClauseRef, ...] = ()

    def derived(
        self,
        *,
        label: str | None = None,
        kind: SpectrumKind | None = None,
        add_parameters: dict[str, float | str] | None = None,
        add_refs: Sequence[ClauseRef] = (),
    ) -> SpectrumMeta:
        """Return a copy describing a transformed spectrum."""
        params = dict(self.parameters)
        if add_parameters:
            params.update(add_parameters)
        return replace(
            self,
            label=self.label if label is None else label,
            kind=self.kind if kind is None else kind,
            parameters=params,
            refs=self.refs + tuple(add_refs),
        )

    @property
    def title(self) -> str:
        """Human-facing one-liner, e.g. ``"ASCE/SEI 7 7-16 design spectrum"``."""
        if self.label:
            return self.label
        head = " ".join(p for p in (self.standard, self.edition) if p)
        return f"{head} {self.kind.value} spectrum".strip()


class Spectrum(ABC):
    """Abstract 5%-damped response spectrum, ordinates in g."""

    __slots__ = ("meta",)

    def __init__(self, meta: SpectrumMeta | None = None) -> None:
        self.meta = meta if meta is not None else SpectrumMeta()

    # -- Evaluation -------------------------------------------------------
    @abstractmethod
    def _evaluate(self, periods: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return Sa (g) at each period in ``periods`` (already validated)."""

    @property
    @abstractmethod
    def t_max(self) -> float:
        """Largest period the spectrum is meaningfully defined to, in seconds."""

    def at(self, period: ArrayLike) -> Any:
        """Spectral acceleration in g at ``period`` (scalar or array, seconds).

        Returns a float for scalar input and an ``ndarray`` otherwise.
        """
        arr = np.atleast_1d(np.asarray(period, dtype=float))
        if np.any(arr < 0.0):
            raise InvalidInput("Period must be non-negative.")
        if not np.all(np.isfinite(arr)):
            raise InvalidInput("Period must be finite.")
        out = self._evaluate(arr)
        if np.isscalar(period) or np.ndim(period) == 0:
            return float(out[0])
        return out

    def __call__(self, period: ArrayLike) -> Any:
        return self.at(period)

    # -- Sampling ---------------------------------------------------------
    @property
    def control_periods(self) -> ControlPeriods:
        """Named breakpoints of this spectrum."""
        return self.meta.control_periods

    def grid(self, n: int = 400, t_max: float | None = None) -> NDArray[np.float64]:
        """A sampling grid that lands exactly on every control period.

        A plain ``linspace`` chamfers the plateau corners of a code spectrum;
        this injects the breakpoints so plotted and exported curves match the
        published figure.
        """
        upper = float(t_max) if t_max is not None else self.t_max
        if upper <= 0.0:
            raise InvalidInput("t_max must be positive.")
        if n < 2:
            raise InvalidInput("n must be at least 2.")
        base = np.linspace(0.0, upper, n)
        return self.control_periods.refine_grid(base)

    def sample(
        self,
        periods: ArrayLike | None = None,
        *,
        n: int = 400,
        t_max: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(T, Sa)`` arrays. Defaults to :meth:`grid`."""
        if periods is None:
            T = self.grid(n=n, t_max=t_max)
        else:
            T = np.atleast_1d(np.asarray(periods, dtype=float))
        return T, np.atleast_1d(self.at(T))

    # -- Transformations --------------------------------------------------
    def _transform(
        self,
        fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        meta: SpectrumMeta,
    ) -> Spectrum:
        """Apply ``fn`` to the ordinates, preserving exactness where possible."""
        raise NotImplementedError  # pragma: no cover - overridden below

    def scaled(self, factor: float, *, label: str | None = None) -> Spectrum:
        """Multiply every ordinate by ``factor``."""
        if factor <= 0.0:
            raise InvalidInput("Scale factor must be positive.")
        meta = self.meta.derived(
            label=label if label is not None else f"{self.meta.title} x {factor:g}",
            add_parameters={"scale_factor": float(factor)},
        )
        return self._transform(lambda sa: sa * factor, meta)

    def reduced(
        self,
        R: float,
        *,
        Ie: float = 1.0,
        phi_p: float = 1.0,
        phi_e: float = 1.0,
        label: str | None = None,
    ) -> Spectrum:
        """Reduce for inelastic response.

        Computes ``Sa * Ie / (R * phi_p * phi_e)``. ASCE 7 uses ``R`` and
        ``Ie``; NEC-SE-DS uses ``R``, ``phi_p`` and ``phi_e`` with importance
        already folded into ``Ie``. Leave the factors you do not need at 1.0.
        """
        for name, value in (("R", R), ("Ie", Ie), ("phi_p", phi_p), ("phi_e", phi_e)):
            if value <= 0.0:
                raise InvalidInput(f"{name} must be positive, got {value}.")
        divisor = R * phi_p * phi_e
        meta = self.meta.derived(
            label=label if label is not None else f"{self.meta.title} reduced (R={R:g})",
            kind=SpectrumKind.INELASTIC,
            add_parameters={"R": float(R), "Ie": float(Ie), "phi_p": float(phi_p),
                            "phi_e": float(phi_e)},
        )
        return self._transform(lambda sa: sa * Ie / divisor, meta)

    def resampled(self, periods: ArrayLike) -> TabulatedSpectrum:
        """Freeze this spectrum onto a fixed period grid."""
        T, Sa = self.sample(periods)
        return TabulatedSpectrum(T, Sa, meta=self.meta)

    def envelope(self, *others: Spectrum, label: str | None = None) -> Spectrum:
        """Point-wise maximum of this spectrum and ``others``."""
        return _combine(
            (self, *others),
            np.maximum,
            label or f"envelope({', '.join(s.meta.title for s in (self, *others))})",
        )

    def floored_by(
        self,
        other: Spectrum,
        ratio: float = 1.0,
        *,
        label: str | None = None,
    ) -> Spectrum:
        """Enforce a lower bound of ``ratio * other`` at every period.

        This is how code floors on site-specific spectra are expressed, e.g.
        the ASCE 7 §21.3 requirement that a site-specific design spectrum not
        fall below 80% of the §11.4 spectrum::

            site_specific.floored_by(code_spectrum, 0.80)
        """
        if ratio <= 0.0:
            raise InvalidInput("ratio must be positive.")
        floor = other if ratio == 1.0 else other.scaled(ratio)
        return _combine(
            (self, floor),
            np.maximum,
            label or f"{self.meta.title} floored at {ratio:.0%} of {other.meta.title}",
        )

    def capped_by(
        self,
        other: Spectrum,
        ratio: float = 1.0,
        *,
        label: str | None = None,
    ) -> Spectrum:
        """Enforce an upper bound of ``ratio * other`` at every period."""
        if ratio <= 0.0:
            raise InvalidInput("ratio must be positive.")
        cap = other if ratio == 1.0 else other.scaled(ratio)
        return _combine(
            (self, cap),
            np.minimum,
            label or f"{self.meta.title} capped at {ratio:.0%} of {other.meta.title}",
        )

    # -- Derived quantities -----------------------------------------------
    def peak(self, *, n: int = 2000) -> tuple[float, float]:
        """Return ``(T, Sa)`` at the maximum ordinate over the sampled range."""
        T, Sa = self.sample(n=n)
        idx = int(np.argmax(Sa))
        return float(T[idx]), float(Sa[idx])

    def displacement(self, period: ArrayLike) -> Any:
        """Spectral displacement in metres, ``Sd = Sa g (T / 2 pi)^2``."""
        from .units import STANDARD_GRAVITY

        T = np.atleast_1d(np.asarray(period, dtype=float))
        Sd = np.atleast_1d(self.at(T)) * STANDARD_GRAVITY * (T / (2.0 * np.pi)) ** 2
        if np.isscalar(period) or np.ndim(period) == 0:
            return float(Sd[0])
        return Sd

    # -- Output -----------------------------------------------------------
    def to_numpy(
        self,
        periods: ArrayLike | None = None,
        *,
        n: int = 400,
        unit: AccelUnit | str = AccelUnit.G,
    ) -> NDArray[np.float64]:
        """Return a ``(N, 2)`` array of ``[T, Sa]``."""
        T, Sa = self.sample(periods, n=n)
        return np.column_stack([T, from_g(Sa, unit)])

    def to_dataframe(
        self,
        periods: ArrayLike | None = None,
        *,
        n: int = 400,
        unit: AccelUnit | str = AccelUnit.G,
    ) -> pd.DataFrame:
        """Return a ``DataFrame`` with ``T`` and ``Sa`` columns."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_dataframe requires pandas: pip install 'codeSpectra[tabular]'"
            ) from exc
        T, Sa = self.sample(periods, n=n)
        unit_enum = AccelUnit(unit)
        return pd.DataFrame({"T": T, f"Sa [{unit_enum.label}]": from_g(Sa, unit_enum)})

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot onto ``ax`` (created if omitted). See :mod:`codeSpectra.plotting`."""
        from ..plotting import plot_spectrum

        return plot_spectrum(self, ax=ax, **kwargs)

    # -- Presentation -----------------------------------------------------
    def __repr__(self) -> str:
        cp = ", ".join(f"{k}={v:g}" for k, v in self.control_periods.items())
        return f"<{type(self).__name__} {self.meta.title!r}{' | ' + cp if cp else ''}>"


class AnalyticSpectrum(Spectrum):
    """A spectrum defined by a closed-form function of period.

    ``at(T)`` is exact — no interpolation error at any period, including
    right at a branch corner.
    """

    __slots__ = ("_evaluator", "_t_max")

    def __init__(
        self,
        evaluator: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        meta: SpectrumMeta | None = None,
        t_max: float = 6.0,
    ) -> None:
        super().__init__(meta)
        if t_max <= 0.0:
            raise InvalidInput("t_max must be positive.")
        self._evaluator = evaluator
        self._t_max = float(t_max)

    @property
    def t_max(self) -> float:
        return self._t_max

    def _evaluate(self, periods: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._evaluator(periods), dtype=float)

    def _transform(
        self,
        fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        meta: SpectrumMeta,
    ) -> Spectrum:
        inner = self._evaluator
        return AnalyticSpectrum(
            lambda T: fn(np.asarray(inner(T), dtype=float)),
            meta=meta,
            t_max=self._t_max,
        )

    def with_t_max(self, t_max: float) -> AnalyticSpectrum:
        """Return the same spectrum reported over a different period range."""
        return AnalyticSpectrum(self._evaluator, meta=self.meta, t_max=t_max)


class TabulatedSpectrum(Spectrum):
    """A spectrum defined by ordinates, interpolated linearly in period.

    Beyond the last tabulated period the ordinate decays as ``T_last/T``, and
    as ``T_last * TL / T^2`` past ``TL`` when a ``TL`` control period is
    present — the ASCE 7-22 §11.4.5.1 rule for periods above 10 s. Below the
    first tabulated period the first ordinate is held.
    """

    __slots__ = ("_Sa", "_T")

    _T: NDArray[np.float64]
    _Sa: NDArray[np.float64]

    def __init__(
        self,
        periods: ArrayLike,
        ordinates: ArrayLike,
        *,
        meta: SpectrumMeta | None = None,
    ) -> None:
        super().__init__(meta)
        T = np.atleast_1d(np.asarray(periods, dtype=float))
        Sa = np.atleast_1d(np.asarray(ordinates, dtype=float))
        if T.ndim != 1 or Sa.ndim != 1:
            raise InvalidInput("periods and ordinates must be one-dimensional.")
        if T.size != Sa.size:
            raise InvalidInput(
                f"periods and ordinates length mismatch: {T.size} vs {Sa.size}."
            )
        if T.size < 2:
            raise InvalidInput("A tabulated spectrum needs at least two points.")
        if np.any(T < 0.0):
            raise InvalidInput("Periods must be non-negative.")
        if np.any(Sa < 0.0):
            raise InvalidInput("Spectral ordinates must be non-negative.")
        order = np.argsort(T, kind="stable")
        T, Sa = T[order], Sa[order]
        if np.any(np.diff(T) <= 0.0):
            raise InvalidInput("Periods must be strictly increasing (duplicates found).")
        self._T = T
        self._Sa = Sa

    @property
    def periods(self) -> NDArray[np.float64]:
        """Tabulated periods, seconds (a copy)."""
        return self._T.copy()

    @property
    def ordinates(self) -> NDArray[np.float64]:
        """Tabulated ordinates, g (a copy)."""
        return self._Sa.copy()

    @property
    def t_max(self) -> float:
        return float(self._T[-1])

    def _evaluate(self, periods: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.interp(periods, self._T, self._Sa)
        t_last, sa_last = float(self._T[-1]), float(self._Sa[-1])
        beyond = periods > t_last
        if np.any(beyond) and t_last > 0.0:
            TL = self.control_periods.get("TL")
            T_b = periods[beyond]
            tail = sa_last * t_last / T_b
            if TL is not None and np.isfinite(TL) and TL > 0.0:
                long_side = T_b > TL
                tail = np.where(long_side, sa_last * t_last * TL / T_b**2, tail)
            out[beyond] = tail
        return np.asarray(out, dtype=np.float64)

    def grid(self, n: int = 400, t_max: float | None = None) -> NDArray[np.float64]:
        """Tabulated periods themselves, plus control periods within range."""
        upper = float(t_max) if t_max is not None else self.t_max
        base = self._T[upper >= self._T]
        if base.size == 0 or base[-1] < upper:
            base = np.append(base, upper)
        return self.control_periods.refine_grid(base)

    def _transform(
        self,
        fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        meta: SpectrumMeta,
    ) -> Spectrum:
        return TabulatedSpectrum(self._T, fn(self._Sa), meta=meta)


def _combine(
    spectra: Sequence[Spectrum],
    op: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    label: str,
) -> AnalyticSpectrum:
    """Point-wise combination of spectra, evaluated lazily and exactly."""
    if not spectra:
        raise InvalidInput("At least one spectrum is required.")
    first = spectra[0]
    merged: dict[str, float] = {}
    for s in spectra:
        merged.update(s.control_periods.as_dict())
    refs: tuple[ClauseRef, ...] = ()
    for s in spectra:
        refs += s.meta.refs
    meta = replace(
        first.meta,
        label=label,
        control_periods=ControlPeriods(**merged),
        refs=tuple(dict.fromkeys(refs)),
    )

    def evaluate(T: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.atleast_1d(spectra[0].at(T)).astype(float)
        for s in spectra[1:]:
            result = op(result, np.atleast_1d(s.at(T)).astype(float))
        return np.asarray(result, dtype=np.float64)

    return AnalyticSpectrum(
        evaluate, meta=meta, t_max=max(s.t_max for s in spectra)
    )
