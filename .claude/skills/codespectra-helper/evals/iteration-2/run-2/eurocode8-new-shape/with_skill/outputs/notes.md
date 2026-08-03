# Notes — EN 1998-1 §3.2.2.2 Type 1 horizontal elastic spectrum

## 0. Environment

- `git reset --hard 198bbad` applied first. Confirmed `src/codeSpectra/codes/`
  and `.claude/skills/codespectra-helper/SKILL.md` both present.
- All Python run with `PYTHONPATH=<worktree>/src`; verified by printing
  `codeSpectra.__file__`, which resolves inside this worktree, not the
  editable install elsewhere on the machine.
- Read `SKILL.md`, `references/adding-a-code.md` and
  `references/pdf-extraction.md` before touching anything.

## 1. What was added

| Path | Contents |
|---|---|
| `src/codeSpectra/codes/en1998/en1998_1.py` | `EN1998_1`, `GroundType`, `damping_correction`, `TYPE_1_PARAMETERS` |
| `src/codeSpectra/codes/en1998/__init__.py` | re-exports |
| `tests/test_en1998.py` | 117 tests |
| `src/codeSpectra/__init__.py` | lazy `__getattr__` entries + `__all__` |
| `README.md`, `pyproject.toml` | standards table, worked example, verification list, "not implemented" list, description/keywords |

Followed `adding-a-code.md` §3–§8: frozen dataclass, validating
`__post_init__`, `cached_property` for the derived tuple, `control_periods`,
a vectorised closed-form evaluator wrapped in `AnalyticSpectrum`, `report()`,
tests in all six categories, wired into both `__init__`s.

## 2. Extraction and how it was validated

Source: `C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos\EC 8\Eurocode 8 Part 1 - prEN 1998-1 (12-2003).pdf`.
Located §3.2.2.2 by probing pages for a content regex (`Se(T)`, `Table 3.2`,
"damping correction") rather than by printed page number — the clause sits on
PDF pages 37–40 (printed 23–26).

**Trap 1 — Table 3.2 columns desync under `-layout`.** Exactly the failure
mode §7 of the skill and §5 of `pdf-extraction.md` warn about. `-layout`
renders the page as:

```
    Ground type       S     TB (s)  TC (s)  TD (s)
                            0,15     0,4     2,0
    A                 1,0   0,15     0,5     2,0
                            0,20     0,6     2,0
    B                 1,2   0,20     0,8     2,0
                            0,15     0,5     2,0
    C                 1,15
```

Reading that naively pairs ground type A with type B's periods. Validation
performed:

1. `pdftotext -table` — clean one-row-per-ground-type output.
2. `pdftotext -raw` — independent mode, agrees row for row with `-table`.
3. `pdftoppm -r 200` render of the page, read visually. The bordered table is
   unambiguous.

All three agree:

| Ground type | S | TB (s) | TC (s) | TD (s) |
|---|---|---|---|---|
| A | 1.0 | 0.15 | 0.4 | 2.0 |
| B | 1.2 | 0.15 | 0.5 | 2.0 |
| C | 1.15 | 0.20 | 0.6 | 2.0 |
| D | 1.35 | 0.20 | 0.8 | 2.0 |
| E | 1.4 | 0.15 | 0.5 | 2.0 |

Also cross-checked structurally: Table 3.3 (Type 2) on the next page exhibits
the *same* `-layout` interleave, and its `-table`/`-raw` reading is internally
consistent, which confirms the interleave interpretation rather than an
accidental match. (Type 2 was read only as a cross-check; it is not shipped.)

**Trap 2 — the radical in Eq. 3.6 is not in the text layer.** `-raw` yields
the reversed-fragment soup `( ) 55 , 0 5 / 10 ≥ + = ξ η`, and `-layout` yields
`η = 10 /(5 + ξ ) ≥ 0,55`. The square-root sign is a drawn glyph. This is
dangerous rather than merely annoying, because **both** `10/(5+ξ)` and
`sqrt(10/(5+ξ))` give η = 1 at ξ = 5%, so the reference-damping sanity check
cannot distinguish them. Resolved by rendering page 40 at 200 dpi and reading
it: `η = √(10/(5+ξ)) ≥ 0,55`. The 0.55 floor was read off the same render.

Non-ASCII audit of the extracted region: `η ξ π ⋅ − ≤ ≥` plus PUA `U+F8EE–
U+F8FB` (Symbol-font large-parenthesis pieces, in Eqs. 3.8–3.11 which are not
used here). No mis-mapped accented characters of the NEC kind. Nothing
transcribed contains non-ASCII.

**Independent numerical check.** With ag = 1.0 g at 5% damping the plateaus
come out at 2.5 / 3.0 / 2.875 / 3.375 / 3.5 for A/B/C/D/E, which matches the
plateau heights and the plateau end-periods in the rendered Figure 3.2
(including that E is the tallest and D's plateau runs longest, and the visible
slope change on D at T = 2 s = TD).

Other clauses read and cited, not guessed:

- §3.2.1(3): `ag = gamma_I · agR` (implemented as the `agR`/`gamma_I` path).
- §3.1.2(4)P and NOTE 2 to §3.2.2.2(2)P: S1/S2 require special studies.
- §3.2.2.2(2)P NOTE 1: S/TB/TC/TD are National-Annex parameters; Type 2 is
  recommended when Ms ≤ 5.5.
- §3.2.2.2(4): non-5% damping is used only where a Part of EN 1998 gives one.
- §3.2.2.2(6): Se(T) applies to periods up to 4.0 s; Annex A beyond.
- §3.2.1(4)/(5) NOTEs: recommended low- and very-low-seismicity thresholds.
- Table 3.1: ground-type stratigraphic descriptions.

## 3. Design decisions

- **`ag` in g.** The library's invariant is Sa in g and periods in seconds
  (SKILL §4). Eurocode prints ag in m/s², so the docstring and README say
  "divide by 9.81". Not introducing a second unit system.
- **`damping` as a percentage.** Eq. 3.6 is written with ξ in per cent; using
  a fraction would silently produce η ≈ 1.41 for a caller who passed 0.05.
  Validated to (0, 100] and the error message names the convention. The
  spectrum label prints `2% damping`, so a fraction mistake shows up as
  `0.05% damping` in every export header.
- **`eta` is not stored on the site.** Damping is a structural property, not a
  site property, so `elastic_spectrum(damping=...)` and `report(damping=...)`
  take it, and `SpectrumMeta.damping` (already a field, as a fraction) carries
  it downstream to the exporters.
- **National Annex escape hatches named after what they do** (SKILL §8):
  `S_override`/`TB_override`/`TC_override`/`TD_override`, not a boolean.
  Using one makes the corresponding report note `notes[0]` — the non-code
  provenance rule from SKILL §5.
- **S1/S2 raise rather than return.** They are members of `GroundType` (they
  are in Table 3.1 and carry descriptions) but constructing an `EN1998_1` with
  either raises `SiteSpecificRequired` carrying both clauses and a remedy.
- **Beyond 4 s.** `t_max` defaults to 4.0 so `grid()`, `sample()` and every
  exporter stay inside the range §3.2.2.2 defines. `at(T)` above 4 s continues
  the Eq. 3.5 form rather than raising (consistent with how the other codes
  behave past their last breakpoint), and a report note states plainly that
  EN 1998-1 instead refers to Annex A, which is not implemented.
- **Scope held to what was asked.** Type 2 (Table 3.3), the vertical spectrum
  §3.2.2.3, the §3.2.2.5 design spectrum `Sd(T)` and Annex A are not
  implemented, and each is named in `report().notes` and the README. `Sd(T)`
  gets an explicit warning that it is *not* `Se(T)/q`, since `Spectrum.reduced`
  would make that mistake easy.
- **Advisory, not blocking**, for the §3.2.1(4)/(5) low-seismicity thresholds:
  both are recommendations the National Annex may replace, and each offers a
  choice between `ag` and `ag·S`, so `low_seismicity_notes` reports rather than
  acts (same pattern as `ASCE7_16.site_specific_triggers`).

## 4. Tests — 117, all six categories

| Category | Tests |
|---|---|
| Table integrity | `TestTable32` — every Table 3.2 cell parametrised against the printed row; ground-type set; control-period names |
| Branch shape | `TestBranchShape` — Se(0), Eq. 3.2 ramp, Eq. 3.3 plateau, Eq. 3.4 1/T, Eq. 3.5 1/T², peak location, finiteness on all five ground types, metadata |
| Continuity | `TestContinuity` — Sa(T±1e-7) at TB, TC, TD for 5 ground types × 5 damping values |
| Code rules | S1/S2 raise `SiteSpecificRequired` with clauses + remedy; `eta` floor at 0.55 and where it starts to govern; ag/agR/gamma_I consistency; override validation; TB ≤ TC ≤ TD |
| Encoding | `report().to_text().encode("cp1252")` for every ground type, including the note-heavy 30%-damping path |
| Provenance | National Annex note is `notes[0]`; NDP warning always present; draft-edition note; Type 2 and §3.2.2.5 documented as unimplemented |

Plus `TestDampingCorrection` (Eq. 3.6 against `math.sqrt` at eight values,
monotonicity, floor, input rejection) and `TestDampingScalesTheWholeCurve`,
which pins the non-obvious behaviour that `Se(0) = ag·S` is damping-independent
because η multiplies only the slope term of Eq. 3.2.

Expected values are hand-computed from the printed expressions, never read back
from the implementation.

## 5. Quality gate — all clean

```
python -m pytest -q                       629 passed (117 new)
python -m ruff check .                    All checks passed!
python -m mypy src/codeSpectra            Success: no issues found in 29 source files
python -c "ast.parse(..., feature_version=(3,10))"   OK
```

Additionally:

- `python -m pytest --doctest-modules src/codeSpectra -q` — 10 passed
  (SKILL §15: doctests don't run in the default suite).
- `python -m build --wheel` — confirmed `codeSpectra/codes/en1998/*.py` reach
  the wheel via `packages.find` auto-discovery. No new `tables/*.json`, so the
  `package-data` glob is unaffected; existing JSONs still land. `dist/`,
  `build/` and the egg-info removed afterwards.
- Export smoke test: `to_etabs` / `to_json` produce headers carrying the
  EN 1998-1 clauses and the damping.

Ruff's `--fix` reordered the new `__all__` / import blocks to its isort
convention; that is the only automated edit.

## 6. Nothing committed

`git status` shows modified `README.md`, `pyproject.toml`,
`src/codeSpectra/__init__.py` and untracked `src/codeSpectra/codes/en1998/`,
`tests/test_en1998.py`. `changes.patch` was produced with `git add -N` on the
untracked paths followed by `git reset` to restore the index.
