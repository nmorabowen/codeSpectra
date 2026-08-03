# Notes — adding EN 1998-1 §3.2.2.2 (Type 1 horizontal elastic spectrum) to codeSpectra

## 0. Checkout

`git reset --hard 198bbad` then `rm -rf .claude/skills`. Confirmed
`src/codeSpectra/codes/` present (asce7, nch433, nec) and `.claude/skills` gone.
All Python run with
`PYTHONPATH=C:/Users/nmora/Documents/GitHub/codeSpectra/.claude/worktrees/agent-a9f26165fb7fbb92b/src`
so the worktree's own source is exercised, not the editable install pointing elsewhere.

## 1. What was built

New package `src/codeSpectra/codes/ec8/`:

| File | Contents |
| --- | --- |
| `_tables.py` | Table 3.2 (Type 1 `S`, `TB`, `TC`, `TD`), Table 3.1 ground descriptions, Table 4.3 `gamma_I`, the `ref()` citation helper, `STANDARD`/`EDITION` |
| `en1998_1.py` | `damping_correction()`, `GroundType`, `ImportanceClass`, `EN1998_1` |
| `__init__.py` | Re-exports |

Wired into `src/codeSpectra/__init__.py` lazy loader (`EN1998_1`, `GroundType`,
`ImportanceClass`). README gets a standards-table row, a worked section, a
verification entry and a not-implemented entry. `pyproject.toml` description and
keywords mention Eurocode 8.

Design follows the existing house style exactly: frozen dataclass with a
`__post_init__` that coerces enums and validates, `cached_property` for the
table lookup, closed-form vectorised evaluator wrapped in `AnalyticSpectrum`,
`ControlPeriods(TB, TC, TD)`, `SpectrumMeta` carrying `ClauseRef`s, and a
`report()` returning `ReportItem`s + notes.

No separate displacement method: Eq. (3.7) `SDe = Se·(T/2π)²` is exactly what
`Spectrum.displacement()` in core already does, so it is documented rather than
duplicated. A test asserts they agree.

## 2. Extraction and how it was validated

Source: `Eurocode 8 Part 1 - prEN 1998-1 (12-2003).pdf` (229 pages).

Text layer via `pdftotext -layout` located §3.2.2.2 and the equations, but the
extraction is lossy in two ways that matter:

* **Symbols are dropped.** Eq. (3.2) came out as `1 + T/TB (2,5 -1)` with `η`
  missing, and Eq. (3.6) as `= 10 /(5 + ) ≥ 0,55` with the radical and `ξ`
  gone. Working from that text alone you would implement the wrong equations.
* **Table 3.2 is row-shifted.** The text layer emits the numeric triples on
  lines *above* their ground-type letters:

  ```
  Ground type   S    TB (s) TC (s) TD (s)
                     0,15   0,4    2,0
  A             1,0  0,15   0,5    2,0
                     0,20   0,6    2,0
  B             1,2  0,20   0,8    2,0
                     0,15   0,5    2,0
  C             1,15
  D             1,35
  E             1,4
  ```

  Read naively, ground type A gets `TB=0.15, TC=0.5` (B's row) and C/D/E get no
  periods at all. This is the dangerous failure mode: it produces a
  self-consistent, plausible table.

**Validation performed:**

1. Rendered pages 34, 37, 38, 39, 40 (0-based PDF indices) at 170 dpi with
   `pdftoppm` and read the tables and equations off the rendered images:
   - p.37 → printed p.23: Eqs. (3.2)–(3.5) and the symbol list. Confirmed `η`
     sits **inside** the bracket of Eq. (3.2), multiplies `2,5` in (3.3)–(3.5),
     and `ag = γI·agR`.
   - p.38 → printed p.24: Figure 3.1 and Table 3.2. Confirmed
     A(1.0, 0.15, 0.4, 2.0), B(1.2, 0.15, 0.5, 2.0), C(1.15, 0.20, 0.6, 2.0),
     D(1.35, 0.20, 0.8, 2.0), E(1.4, 0.15, 0.5, 2.0).
   - p.40 → printed p.26: Eq. (3.6) `η = √(10/(5+ξ)) ≥ 0.55`, `ξ` in percent;
     Eq. (3.7); §3.2.2.2(6) 4 s validity limit.
   - p.34 → printed p.20: Table 3.1 ground descriptions incl. S1/S2.
2. **Independent cross-check against Figure 3.2** (printed p.25, the published
   normalised Type 1 curves at 5% damping). Ran the implementation with
   `ag = 1.0` g and compared:

   | | A | B | C | D | E |
   |---|---|---|---|---|---|
   | `Se(0)/ag` computed | 1.000 | 1.200 | 1.150 | 1.350 | 1.400 |
   | plateau computed | 2.500 | 3.000 | 2.875 | 3.375 | 3.500 |
   | plateau ends at (s) | 0.4 | 0.5 | 0.6 | 0.8 | 0.5 |

   All three rows match the printed figure, including the non-monotonic
   ordering (E highest at 3.5, D at 3.375, B at 3.0 above C at 2.875). A
   row-shifted transcription would give a monotonic `S` column and would not
   reproduce the figure.
3. **Table 3.3 as a corroborating check.** The same row-shift reading applied to
   Table 3.3 (Type 2) yields A(1.0, 0.05, 0.25, 1.2), B(1.35, 0.05, 0.25, 1.2),
   C(1.5, 0.10, 0.25, 1.2), D(1.8, 0.10, 0.30, 1.2), E(1.6, 0.05, 0.25, 1.2),
   which is exactly what the rendered p.39 shows. That the same de-shift rule
   works on both tables confirms the reading rule rather than a lucky guess.
   (Type 2 itself is *not* implemented — out of the requested scope.)
4. `gamma_I` recommended values 0.8 / 1.0 / 1.2 / 1.4 read from the 4.2.5(5)P
   NOTE at line 2951 of the text extract, cross-read against the surrounding
   paragraph. Class II = 1.0 is normative ("by definition"); the other three are
   recommendations, flagged as NDPs in code and report.

Nothing was taken from memory. `S`, `TB`, `TC`, `TD`, `gamma_I` and the ground
descriptions all trace to a page I rendered and read.

## 3. Judgement calls

* **Percent-vs-fraction guard.** Eq. (3.6) takes `ξ` in percent. Passing `0.05`
  for 5% returns `η = 1.407` — a silent 41% overestimate of the entire
  spectrum, the worst kind of bug because the result still looks like a
  spectrum. `damping_correction()` therefore raises `InvalidInput` for any value
  in `(0, 1)` with a message naming the likely intent. This blocks sub-1%
  damping, which is essentially never used in EN 1998 building design; the
  docstring and the error message both say the guard is deliberate and point at
  the override route.
* **`η` inside the ascending bracket.** Implemented per Eq. (3.2) so
  `Se(0) = ag·S` regardless of damping. Two tests lock this: one asserting the
  zero-period ordinate is damping-invariant, one asserting the below-`TB` ratio
  lies strictly between 1 and `η` (i.e. the branch is *not* uniformly scaled).
* **S1 / S2 raise rather than invent.** 3.2.2.2(2)P NOTE 2 requires special
  studies. Follows the repo's existing convention (`SiteSpecificRequired` with
  refs + remedy, matching NEC soil F). All four overrides must be supplied; a
  partial override still raises, naming the missing parameters.
* **Overrides double as the National Annex hook.** Every Table 3.2 value is an
  NDP. Rather than pretend to ship national parameter sets, the four overrides
  serve both S1/S2 studies and National Annex departures, and a report note
  records when a value was user-supplied.
* **Edition string is `prEN 1998-1:2003`, not `EN 1998-1:2004`.** The PDF read
  is the December 2003 final draft. Citing the published edition would be
  asserting something not verified from a source in hand. A report note names
  the draft as the transcription source and asks the reader to check against the
  edition in force. Flagged in the user-facing reply too.
* **Scope held to Type 1.** The request named Type 1 explicitly. Type 2, the
  vertical spectrum, and the §3.2.2.5 `q`-reduced design spectrum are listed as
  not implemented in the README and in a report note on every instance, so the
  gap is visible at the point of use rather than only in the docs.
* **`t_max` defaults to 4.0 s** per §3.2.2.2(6). Eq. (3.5) still evaluates past
  4 s if a caller asks; the limit and the Annex A pointer are in a report note.
* **Validation of `TB < TC < TD`** after overrides are merged, so a bad
  National Annex entry fails loudly at construction.

## 4. Quality gate

Run with the worktree on `PYTHONPATH`.

| Check | Result |
| --- | --- |
| `python -m pytest` | **610 passed** (512 pre-existing + 98 new), 0.85 s |
| `python -m pytest tests/test_ec8.py` | 98 passed |
| `python -m ruff check src tests` | All checks passed |
| `python -m mypy src` (strict) | Success: no issues found in 30 source files |
| `python -m pytest --doctest-modules src/codeSpectra` | 10 passed (incl. 3 new docstring examples) |
| Figure 3.2 cross-check | plateaus and corner periods match for all five ground types |
| `report().to_text().encode("cp1252")` | passes (repo convention for Windows console safety) |

Two fixes made during the gate: one `ruff` `RUF022` `__all__` sort (auto-fixed)
and four `mypy` strict errors from a `dict[str, float | None]` merge, fixed by
building the merged parameter dict with an explicit loop that also collects the
missing-key list used by the `SiteSpecificRequired` message.

No commits made. `git add -N` was used to include the new files in the diff and
the index was reset afterwards; the emitted patch is scoped to
`src tests README.md pyproject.toml` so the STEP 0 `.claude/skills` deletion does
not pollute it.
