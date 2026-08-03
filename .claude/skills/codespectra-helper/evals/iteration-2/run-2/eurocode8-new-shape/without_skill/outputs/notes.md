# Notes — EN 1998-1 §3.2.2.2 horizontal elastic response spectrum

## Setup

Ran the prescribed `git reset --hard 198bbad` and `rm -rf .claude/skills`.
Confirmed `src/codeSpectra/codes/` exists (`asce7`, `nch433`, `nec`) and
`.claude/skills` is gone (`.claude/` is now empty). All Python was run with
`PYTHONPATH=<worktree>/src` so the local source was exercised, not the
`pip install -e` pointing at a different worktree.

## Files

| Path | What |
| --- | --- |
| `src/codeSpectra/codes/eurocode8/en1998_1.py` | New. `EN1998_1`, `GroundType`, `SpectrumShape`, `damping_correction`, Tables 3.1/3.2/3.3. |
| `src/codeSpectra/codes/eurocode8/__init__.py` | New. Subpackage exports. |
| `tests/test_eurocode8.py` | New. 141 tests. |
| `src/codeSpectra/__init__.py` | `EN1998_1`, `GroundType`, `SpectrumShape` added to the lazy `__getattr__` registry and `__all__`. |
| `README.md` | Standards table row, a Eurocode 8 section, verification bullet, "Not implemented" bullet. |
| `pyproject.toml` | Description + keywords. |

Nothing committed. `changes.patch` is `git diff -- src tests README.md
pyproject.toml` after `git add -N` on the two new paths, index restored with
`git reset`. **The `.claude/skills` deletion is excluded from the patch on
purpose** — it was an instructed setup step, not part of this change, and
1,400 lines of deleted eval fixtures would bury the actual diff.

## Extraction and how it was validated

Source: `C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos\EC 8\
Eurocode 8 Part 1 - prEN 1998-1 (12-2003).pdf`, PDF pages 37–40 = printed
pages 23–26.

Two extraction traps hit here, both silent:

1. **Table 3.2 column drift.** `pdftotext -layout` emits the `S` column one
   line *below* its `TB`/`TC`/`TD` row:

   ```
   Ground type       S     TB (s)  TC (s)  TD (s)
                           0,15     0,4     2,0
   A                 1,0   0,15     0,5     2,0
                           0,20     0,6     2,0
   B                 1,2   0,20     0,8     2,0
                           0,15     0,5     2,0
   C                 1,15
   ```

   Read naively, ground type A gets `TB/TC/TD = 0.15/0.5/2.0` (which is B's
   row) and E gets nothing. The `S` values are in the right order; only the
   period rows are offset by one line.

2. **Eq. (3.6) loses its radical.** Text extraction gives
   `η = 10 /(5 + ξ) ≥ 0,55` — the √ glyph and the η/ξ/≥ symbols are Symbol-font
   characters that drop out. Implementing that literally gives η(2%) = 1.43
   instead of 1.20. Same class of loss in Eq. (3.2)–(3.5): the η symbol
   disappears from every one of them.

**Validation.** Rendered PDF pages 37–40 at 160 dpi with `pdftoppm` and read
the tables, all five equations, and the Figure 3.1/3.2/3.3 shapes off the
images. That confirmed:

- Table 3.2 (Type 1): A 1.0/0.15/0.4/2.0 · B 1.2/0.15/0.5/2.0 ·
  C 1.15/0.20/0.6/2.0 · D 1.35/0.20/0.8/2.0 · E 1.4/0.15/0.5/2.0
- Table 3.3 (Type 2): A 1.0/0.05/0.25/1.2 · B 1.35/0.05/0.25/1.2 ·
  C 1.5/0.10/0.25/1.2 · D 1.8/0.10/0.30/1.2 · E 1.6/0.05/0.25/1.2
- `η = √(10/(5+ξ)) ≥ 0,55`, ξ in percent
- Eq. (3.2)–(3.5) with η multiplying the 2.5 in all four

**Independent cross-check against the published figures.** With `ag = 1`,
5% damping, the implementation returns plateau `Se/ag`:

- Type 1 (Fig. 3.2): A 2.500, B 3.000, C 2.875, D 3.375, E 3.500
- Type 2 (Fig. 3.3): A 2.500, B 3.375, C 3.750, D 4.500, E 4.000

Both match `2.5·S` and reproduce the curve ordering annotated on the figures
(Fig. 3.2 top-to-bottom E, D, C, B, A; Fig. 3.3 D, E, C, B, A). This is a real
check of the table transcription, not a restatement of it: if the Table 3.2
column drift had gone unnoticed, C's peak would be 3.0 and the ordering would
break.

## Design decisions

**Damping as a fraction, with a hard reject on percentages.** Eq. (3.6) wants
ξ in percent; `SpectrumMeta.damping` and the rest of the library use fractions.
Converting inside and raising `InvalidInput` on `damping >= 1.0` closes the
5-vs-0.05 hole. Left open, `eta(5)` returns exactly 1.0 — a plausible-looking
number for a nonsense input, and the worst kind of bug because it produces the
5%-damped curve you were probably comparing against.

**η does not multiply `Se(0)`.** Eq. (3.2) is
`ag·S·[1 + (T/TB)(η·2.5 − 1)]`, so at T = 0 the bracket is 1 and
`Se(0) = ag·S` regardless of damping. Factoring η out front — the tempting
simplification — would wrongly scale the intercept. Covered by
`test_ramp_start_is_damping_independent` and
`test_ordinate_at_zero_is_ag_S`.

**Tables 3.2/3.3 treated as informative.** They come from NOTE 1 to
§3.2.2.2(2)P, which explicitly defers to the National Annex. Each of S, TB, TC,
TD has an `*_override`, `overridden` reports which are in use, and `report()`
carries a permanent note plus a leading flag when overrides are active. No
National Annex values are bundled — that would be data this repo cannot source
from the standard in hand.

**S1/S2 raise `SiteSpecificRequired`**, matching the library's stated principle
(NEC soil F, ASCE Site Class F). Note 2 says special studies must provide the
parameters. Escape hatch: supply all four overrides and it proceeds — a partial
set still raises, since a half-specified special-ground spectrum is worse than
none.

**Type 2 included.** Beyond the literal ask of "Type 1", but it is the same
clause, the same four equations, and the facing page of the same PDF; the
extraction and verification were a single pass. Type 1 remains the default.
Judged useful rather than gold-plating because the shape choice is
hazard-driven (`Ms ≤ 5.5`) and a European user will meet it.

**`t_max = 4.0`.** Eq. (3.5) is written `TD ≤ T ≤ 4s`, and §3.2.2.2(6) sends
longer periods to the Annex A displacement spectrum. `at(T)` past 4 s keeps
applying Eq. (3.5) (the universal practical convention) but `t_max` stops
sampling, plotting and export there, and the report states the limit.

**Scope held to §3.2.2.2.** Not implemented, and listed in the README: §3.2.2.3
vertical spectrum, §3.2.2.4 `dg`, §3.2.2.5 design spectrum with the behaviour
factor `q`, Annex A.

## Quality gate

| Check | Result |
| --- | --- |
| `python -m pytest` | **653 passed** (141 new in `tests/test_eurocode8.py`) |
| `python -m pytest --doctest-modules src/codeSpectra/codes/eurocode8` | **2 passed** |
| `python -m ruff check src tests` | **All checks passed** |
| `python -m mypy src tests` | 10 errors — **identical to the pre-change baseline** (verified by `git stash`; all in `tests/test_export.py`, `tests/test_core.py`, `tests/test_nec_hazard.py`, none in files touched here) |

Test coverage of the new module: every cell of Tables 3.2 and 3.3; Eq. (3.2)–
(3.5) each against its own algebra; Eq. (3.6) at six damping values plus the
0.55 floor and the ξ ≈ 28% engagement point; continuity at TB/TC/TD across
5 ground types × 2 shapes × 3 damping levels; peak = plateau and peak located
in [TB, TC]; linearity in `ag`; `gamma_I`; the Figure 3.2 `2.5·S` ratios; the
percentage-vs-fraction reject on both `damping_correction` and
`elastic_spectrum`; S1/S2 site-specific behaviour with and without overrides;
override ordering validation (TB ≤ TC ≤ TD); report citations and notes;
cp1252-safety of the rendered report; top-level import.
