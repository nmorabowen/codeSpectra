# Notes — EN 1998-1 §3.2.2.2 Type 1 horizontal elastic spectrum

## Setup

`git reset --hard 198bbad` as instructed. Confirmed `src/codeSpectra/codes/`
(asce7, nec, nch433) and `.claude/skills/codespectra-helper/SKILL.md` both
present. Every Python invocation used `PYTHONPATH=src` from this worktree, so
the editable install pointing at a different worktree was never exercised
(SKILL §15).

Read `SKILL.md`, then `references/adding-a-code.md` and
`references/pdf-extraction.md` before touching the PDF.

## What was built

| File | Role |
|---|---|
| `src/codeSpectra/codes/en1998/en1998_1.py` | new — `EN1998_1`, `GroundType`, `ImportanceClass`, `damping_correction` |
| `src/codeSpectra/codes/en1998/__init__.py` | new — family re-export |
| `tests/test_en1998.py` | new — 99 tests |
| `src/codeSpectra/__init__.py` | lazy `__getattr__` map + `__all__` |
| `README.md` | standards table, usage section, verification list, not-implemented list |

Scope held to what was asked: **Type 1 horizontal elastic spectrum only**,
ground types A–E, with η. Type 2 (Table 3.3), the vertical spectrum
(§3.2.2.3) and the §3.2.2.5 design spectrum are stated as not implemented in
the report notes, the README and the reply, rather than half-built.

## Extraction and how it was validated

Source: `Eurocode 8 Part 1 - prEN 1998-1 (12-2003).pdf`, 229 pages.
Located §3.2.2.2 by probing for `Ground type` / `TB (s)` rather than printed
page numbers (SKILL §7) — it lands on **PDF pages 37–40** (printed 23–26),
not where the printed numbering would suggest.

`pdftotext` here is Xpdf 4.00, as the playbook says.

### Table 3.2 — four independent confirmations

1. **`-layout`** — *scrambled*. It emits the `S` column offset by one row from
   the period columns, so a naive read gives ground type C as
   `S=1.15, TB=0.15, TC=0.5, TD=2.0`. This is precisely the multi-column
   desync the playbook warns about for ASCE 7-16 Table 11.9-1.
2. **`-table`** — clean and aligned, five rows.
3. **raw (no flag)** — emits the table column-major
   (`TB (s) 0,15 0,15 0,20 0,20 0,15`), an *independent* ordering that
   confirms the row/column assignment rather than merely repeating it.
4. **200-dpi page image of PDF page 38**, read visually.

All four agree on:

| Ground type | S | TB (s) | TC (s) | TD (s) |
|---|---|---|---|---|
| A | 1,0 | 0,15 | 0,4 | 2,0 |
| B | 1,2 | 0,15 | 0,5 | 2,0 |
| C | 1,15 | 0,20 | 0,6 | 2,0 |
| D | 1,35 | 0,20 | 0,8 | 2,0 |
| E | 1,4 | 0,15 | 0,5 | 2,0 |

Validation checks run (SKILL §7): row count = 5; the S column is
non-monotonic (C = 1,15 sits *below* B = 1,20) and that inversion survives
in all four renders, which is a strong desync detector; TD is 2,0 in every
row; no empty cells; no non-ASCII in the numeric fields. `test_soil_factor_
ordering_matches_the_printed_table` and `test_TD_is_two_seconds_for_every_
type_1_ground` encode the two structural checks so a future re-extraction
cannot regress silently.

### The real find: Eq. (3.6) loses its radical

Every text mode renders

```
η = 10 /(5 + ξ ) ≥ 0,55        (3.6)
```

The published expression is `η = √(10/(5+ξ)) ≥ 0,55`. Confirmed against a
200-dpi render of PDF page 40 (printed 26), where the radical is
unambiguous.

Why this is dangerous and why the usual sanity check misses it: **both forms
give η = 1 at the reference ξ = 5%**, so checking the reference value proves
nothing. The consequences of getting it wrong:

| ξ | correct η | un-rooted η | error |
|---|---|---|---|
| 2% | 1.1952 | 1.4286 | +19.5% on the whole spectrum |
| 7% | 0.9129 | 0.8333 | −8.7% |
| floor bites at | 28.06% | 13.18% | — |

Since the user explicitly asked for 2% and 7% curves, this was the load-bearing
number in the whole task. `test_square_root_is_not_dropped` asserts
η(2%) ≈ 1.1952286 *and* explicitly asserts it is not 10/7.

### Other clauses read from the source, not memory

- §3.2.2.2(1)P Eqs. (3.2)–(3.5) — read off the page image, all four branches.
- §3.2.1(3) `ag = γI·agR`.
- §4.2.5(5)P NOTE — recommended γI = 0.8 / 1.0 / 1.2 / 1.4 for classes I–IV,
  flagged as a Nationally Determined Parameter.
- Table 3.1 ground-type descriptions incl. S1 and S2.
- NOTE 2 to §3.2.2.2(2)P — S1/S2 defer to a special study.
- §3.2.2.2(5)P Eq. (3.7) and (6)'s 4,0 s limit.

## Design decisions

**`damping` on the dataclass, not on `elastic_spectrum()`.** The obvious API is
`elastic_spectrum(damping=0.02)`, but then `site.report()` still prints ξ = 5%
and the two disagree — a silent-wrong-number failure mode of exactly the kind
this library exists to prevent. One source of truth plus a value-object
`with_damping()` derive. `SpectrumMeta.damping` (which already existed) records
what was used, and the spectrum label carries it too.

**Damping as a fraction, with the percentage mistake caught.** EN writes
Eq. (3.6) with ξ as a percentage; the library's convention is fractions
(`SpectrumMeta.damping = 0.05`). Converting inside `damping_correction` keeps
one convention. `damping=2.0` (meaning 2%) would otherwise silently return
η = 0.22 — it raises instead, and the message names the fix.

**S1/S2 raise at lookup, not at construction.** NCh433 rejects soil F in
`__post_init__`, but here NOTE 2 explicitly contemplates a study supplying the
values, so raising at construction would block the legitimate path.
`SiteSpecificRequired` names the missing parameters and the remedy; a complete
set of four overrides proceeds; a partial set still raises.

**National Annex overrides are first-class, not an afterthought.** Unlike ASCE
or NEC, essentially every number in EC8 §3.2.2.2 is a Nationally Determined
Parameter. The `*_override` arguments follow the SKILL §8 convention (named
for what they do), and `is_national_annex_modified` lets a caller detect it.

**No `displacement_spectrum()` builder.** Eq. (3.7) is exactly the existing
`Spectrum.displacement()`. Per SKILL §3 — a new code requirement is usually an
operation on a spectrum, not new code — I added a test asserting they agree
and documented the equivalence, rather than duplicating it.

**Edition string is `prEN 1998-1:2003`, not `EN 1998-1:2004`.** The PDF is the
December 2003 draft. I did not verify the published version, so citing it would
assert something I did not check. The leading report note says so.

**`t_max = 4.0`** — the range Eq. (3.5) is written for.

Branch evaluator follows the `adding-a-code.md` recipe: pre-fill with the
plateau, overwrite the other three branches so no period is unassigned, and
use a `safe` denominator so T = 0 cannot produce `inf` in a masked-out branch.

## Tests — all six SKILL §11 categories

| Category | Tests |
|---|---|
| Table integrity | `TestTable32` — every Table 3.2 cell, γI, structural invariants |
| Branch shape | `TestBranches` — Se(0), ramp linearity, plateau, TC/T decay, TC·TD/T² decay, t_max |
| Continuity | `TestContinuity` — 5 ground types × 4 damping ratios at TB, TC, TD |
| Code rules | `TestCodeRules` — S1/S2 raise, partial study raises, NA overrides, range validation |
| Encoding | `TestReport::test_report_is_cp1252_safe` across all ground types |
| Provenance | draft note is `notes[0]`; NA caveat; Type 2 / §3.2.2.5 gaps documented |

Plus `TestDampingCorrection` (Eq. 3.6 incl. the radical regression and the
0,55 floor at 28% vs 29%), `TestDampingScalesTheWholeCurve`,
`TestImportanceFactor`, `TestDisplacement` (Eq. 3.7), `TestIntegration`.

One test failed on first run — `test_eta_floor_is_reported_when_it_governs`
matched "lower bound" which also occurs in the unrelated §3.2.2.5 note. The
*test* was wrong, not the code (SKILL §11); tightened to match
"Eq. (3.6) lower bound".

Hand-verified ordinates for `agR=0.25, ground C, ξ=2%` against the equations
independently of the module: Se(0)=0.28750, Se(0.10)=0.57329,
Se(0.20)=Se(0.60)=0.85907, Se(1.0)=0.51544, Se(2.0)=0.25772, Se(4.0)=0.06443 g.
All match, and the TD value agrees from both adjacent branches.

## Quality gate (SKILL §10) — all clean

```
python -m pytest -q                    611 passed  (99 new)
python -m ruff check .                 All checks passed!
python -m mypy src/codeSpectra         Success: no issues found in 29 source files
python -c "...ast.parse feature_version=(3,10)..."   py3.10 parse OK
```

Extras: `pytest --doctest-modules src/codeSpectra` passes (SKILL §15 — doctests
are not in the default suite; one docstring value I had hand-computed was wrong
by 1.5e-4 and this caught it). `python -m build --wheel` confirms
`codeSpectra/codes/en1998/*` reaches the wheel; build artifacts removed
afterwards.

Two ruff findings were fixed: `__all__` ordering (autofix) and RUF043
(unescaped `.` in a `pytest.raises(match=)` pattern).

Nothing committed. Working tree: 2 modified, 2 new paths.
