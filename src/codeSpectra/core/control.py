"""Named period breakpoints that delimit a spectrum's branches.

Every code names these differently (ASCE: ``T0``/``Ts``/``TL``; NEC: ``T0``/
``Tc``/``TL``; NCh433: ``To``/``T*``), so this is an ordered, immutable,
attribute-accessible mapping rather than a fixed set of fields.

Its real job is plotting and sampling: a naive ``linspace`` over a code
spectrum rounds off the plateau corners. :meth:`ControlPeriods.refine_grid`
injects the exact breakpoints (and points either side of them) so sampled
curves reproduce the published figure.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

__all__ = ["ControlPeriods"]

#: Offset used to straddle a breakpoint so slope discontinuities render sharply.
_EPS = 1e-9


class ControlPeriods(Mapping[str, float]):
    """Immutable ordered mapping of breakpoint name to period, in seconds.

    Examples
    --------
    >>> cp = ControlPeriods(T0=0.12, Ts=0.60, TL=8.0)
    >>> cp.Ts
    0.6
    >>> cp["T0"]
    0.12
    >>> list(cp)
    ['T0', 'Ts', 'TL']
    """

    __slots__ = ("_data",)

    # Annotated (without assignment) so the catch-all __getattr__ below does
    # not make type checkers infer `_data` as a float.
    _data: Mapping[str, float]

    def __init__(self, **periods: float) -> None:
        object.__setattr__(
            self, "_data", MappingProxyType({k: float(v) for k, v in periods.items()})
        )

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Attribute access -------------------------------------------------
    def __getattr__(self, name: str) -> float:
        # Guard against recursion when _data is absent (copy/pickle protocols).
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no period {name!r}; defined: {list(self._data)}"
            ) from None

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    # -- Presentation -----------------------------------------------------
    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v:g}" for k, v in self._data.items())
        return f"ControlPeriods({body})"

    def as_dict(self) -> dict[str, float]:
        """Return a plain mutable copy."""
        return dict(self._data)

    # -- Grid construction ------------------------------------------------
    def refine_grid(
        self,
        base: NDArray[np.float64],
        *,
        straddle: bool = True,
    ) -> NDArray[np.float64]:
        """Merge the breakpoints into ``base`` and return a sorted unique grid.

        Parameters
        ----------
        base
            Any starting grid of periods, in seconds.
        straddle
            Also insert points a hair either side of each breakpoint, so a
            slope discontinuity plots as a corner rather than a chamfer.
        """
        extra: list[float] = []
        t_max = float(np.max(base)) if base.size else 0.0
        for value in self._data.values():
            if not np.isfinite(value) or value <= 0.0 or value > t_max:
                continue
            extra.append(value)
            if straddle:
                extra.extend((max(value - _EPS, 0.0), value + _EPS))
        if not extra:
            return np.unique(base)
        return np.unique(np.concatenate([base, np.asarray(extra, dtype=float)]))
