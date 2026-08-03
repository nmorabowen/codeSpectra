# Adding a standard or edition to codeSpectra

A worked recipe, using the shape every existing code module follows. Read
`pdf-extraction.md` first if the tables aren't transcribed yet.

## Contents

1. [Decide what actually differs](#1-decide-what-actually-differs)
2. [Tables](#2-tables)
3. [The site class](#3-the-site-class)
4. [The spectrum](#4-the-spectrum)
5. [Deferred values](#5-deferred-values)
6. [Reports](#6-reports)
7. [Tests](#7-tests)
8. [Wiring it up](#8-wiring-it-up)

---

## 1. Decide what actually differs

Before writing anything, work out which of these the new code changes:

- **The spectrum shape.** If it is the ASCE two-period shape, reuse
  `codes.asce7._shared.two_period_spectrum` and supply parameters. NEC and
  NCh433 have different shapes and own their evaluators.
- **The input contract.** What does the user supply, and what does the code
  derive? This is where editions diverge most (ASCE 7-22 deleted `Fa`/`Fv`).
- **The reduction rule.** `Spectrum.reduced(R, Ie, phi_p, phi_e)` already
  covers `Sa·Ie/(R·φP·φE)`. NCh433 needed a period-dependent scalar computed
  first, then a plain scale — not a new mechanism.

If only the tables change, a new edition is mostly a new tables module plus a
thin class. `asce7_10.py` is ~230 lines for exactly this reason.

## 2. Tables

Small coefficient tables become `InterpolatedTable`s next to their citation:

```python
FA_TABLE = InterpolatedTable(
    name="Fa",
    row_label="site class",
    col_label="Ss",
    columns=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    rows={
        "A": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        "E": (2.4, 1.7, 1.3, None, None, None),   # None = §11.4.8 defers
        "F": (None,) * 6,
    },
    ref=ref(EDITION, "11.4.4", "Short-period site coefficient", table="11.4-1"),
    site_specific_remedy="Perform a ground motion hazard analysis per §21.2, or ...",
)
```

Columns are the printed breakpoints; values outside are clamped, because the
standards bound their first and last column with `≤` and `≥`. Interpolation
is straight-line, as the tables instruct.

## 3. The site class

A frozen dataclass, no slots (so `cached_property` works):

```python
@dataclass(frozen=True)
class ASCE7_16:
    Ss: float
    S1: float
    site_class: SiteClass | str = SiteClass.D
    TL: float = 8.0
    risk_category: RiskCategory | str = RiskCategory.II
    Fa_override: float | None = None      # explicit, named for what it does
```

Conventions worth keeping:

- **Coerce enums in `__post_init__`** via `object.__setattr__`, then validate
  ranges and cross-field consistency (`S1 > Ss` is a data-entry error worth
  catching).
- **Expose the computed coefficient under the plain name** (`site.Fa`), and
  name the input override `Fa_override`. An earlier draft used `Fa` for the
  override, so `site.Fa` returned `None` unless you had overridden it — a
  readability trap.
- **`cached_property` for anything derived**, so a report can be rendered
  without re-deriving.
- Add a `sc` (or equivalent) property returning the plain string, to avoid
  `SiteClass(self.site_class).value` scattered everywhere.

## 4. The spectrum

Write the evaluator as a vectorised closed form over a numpy array, and wrap
it in `AnalyticSpectrum`:

```python
def evaluate(T: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.asarray(T, dtype=float)
    out = np.full(T.shape, plateau, dtype=float)
    decay = T > Tc
    safe = np.where(T > 0.0, T, 1.0)        # guard division at T = 0
    out[decay] = plateau * (Tc / safe[decay]) ** r
    return out
```

Two habits that avoid whole classes of bug: build the array pre-filled with
the plateau value and overwrite the branches (so no period is ever
unassigned), and use a `safe` denominator so `T = 0` cannot produce `inf`
even in a branch that will be masked out.

Populate `SpectrumMeta` with `standard`, `edition`, `kind`,
`control_periods`, `parameters` and `refs` — the export and plotting layers
read all of it.

## 5. Deferred values

Wherever the standard says "a site-specific study is required", the lookup
must raise rather than return. `InterpolatedTable` does this for `None`
cells. For rules outside a table, raise `SiteSpecificRequired` directly with
the clause and a `remedy` string.

Where a code exception permits proceeding, implement *that specific
exception* behind a named flag, and pin its limits in a test. ASCE 7-16
§11.4.8 Exception 1 substitutes the Site Class C `Fa` — it does not rescue
`Fv`, and `test_exception_1_does_not_rescue_Fv` exists to stop someone
"fixing" that.

Advisory triggers that have exceptions the library cannot evaluate should be
*reported*, not raised — see `ASCE7_16.site_specific_triggers`.

## 6. Reports

```python
def report(self) -> Report:
    return Report(
        title=f"ASCE/SEI 7-16 seismic ground motion - Site Class {self.sc}",
        items=(ReportItem("SDS", self.SDS, "(2/3) SMS", "g", _R["SDS"]), ...),
        notes=tuple(notes),
    )
```

Keep the text cp1252-safe (§5 of the parent skill) and put any non-code
provenance note first.

## 7. Tests

Mirror the six categories in §11 of the parent skill. The two that catch the
most:

**Every transcribed cell, parametrised:**

```python
@pytest.mark.parametrize(("site_class", "Ss", "expected"), [
    ("C", 0.25, 1.3), ("D", 1.0, 1.1), ("E", 0.75, 1.3),
])
def test_fa(self, site_class, Ss, expected):
    assert FA_TABLE.lookup(site_class, Ss) == pytest.approx(expected)
```

**Continuity at every control period, across every combination:**

```python
@pytest.mark.parametrize("zone", [...])
@pytest.mark.parametrize("soil", [...])
def test_continuous(self, zone, soil):
    s = Site(zone=zone, soil=soil).elastic_spectrum()
    for T in s.control_periods.values():
        assert s.at(T - 1e-7) == pytest.approx(s.at(T + 1e-7), rel=1e-5)
```

A discontinuity means a branch condition is wrong — `<` where `<=` belongs,
or an overlapping mask. This test has found real errors.

## 8. Wiring it up

- Re-export from `codes/<family>/__init__.py`.
- Add to the lazy map in the top-level `codeSpectra/__init__.py` `__getattr__`
  so `from codeSpectra import NewCode` works without making import expensive.
- If you added `tables/*.json`, confirm it lands in a built wheel.
- Update the README's standards table and the "Not implemented" list.
- Run the full four-part gate (§10 of the parent skill).
