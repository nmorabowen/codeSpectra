---
name: codespectra-helper
description: Use whenever working on or with codeSpectra — the Python library producing codified seismic design response spectra per ASCE/SEI 7-10/7-16/7-22, NEC-SE-DS 2015 (Ecuador) and NCh433 (Chile). Triggers on building a design or MCEr spectrum from site parameters, Ss/S1/SDS/SD1/Fa/Fv/TL/T0/Ts, NEC Z/Fa/Fd/Fs/eta/Tc/Tabla 19 poblaciones, NCh433 alpha/R*, the Spectrum value object (AnalyticSpectrum, TabulatedSpectrum, scaled/reduced/envelope/floored_by), ControlPeriods, ClauseRef and Report, equivalent lateral force base shear (ASCE §12.8 Cs/Ta/k, NEC §6.3 V=I·Sa·W/(R·φP·φE)), exporting spectra to ETABS/SAP2000/OpenSees, SiteSpecificRequired, or the Ecuador hazard/gazetteer loaders. Also use it before transcribing ANY design-code table out of a PDF into this repo, and before adding a new standard or edition — the extraction traps and the no-invented-values rule are recorded here and are easy to get wrong. Reach for it even when the user just says "add zone X", "why is Sa wrong at T=0", or "wire in code Y", without naming codeSpectra.
---

# codeSpectra helper

`codeSpectra` turns published seismic design provisions into first-class
Python objects. A spectrum is a value you can scale, reduce, envelope, floor,
export and plot — and every number it produces carries the clause that
authorises it.

Repo: `C:\Users\nmora\Documents\GitHub\codeSpectra` (src-layout,
`src/codeSpectra/`). Package name stays **camelCase** to match the
`ape*` family — this is deliberate, not an oversight.

This skill teaches the repo's **contracts and traps**. For the engineering
theory behind the provisions, reach for `asce7-ground-motion`,
`asce7-seismic-demands` or `asce-seismic`.

---

## 1. The prime directive

> **Never invent a code table value.** Every coefficient in this library is
> transcribed from the standard and asserted in a test. Where a standard
> defers to a site-specific study, the library raises instead of returning a
> plausible number.

This is the whole reason the library exists. An engineer can check a cited
number against the code book; they cannot check a number that was
confabulated. If you cannot find a value in the source PDF, the correct
outcomes are, in order: read the PDF properly, raise `SiteSpecificRequired`
or `InvalidInput`, or tell the user it is not implemented. Guessing is never
one of them.

Corollary: **validate every extraction**. Two separate silent data losses
were caught in this repo only because the extraction asserted on row counts
and leftovers rather than trusting itself (§7).

---

## 2. Architecture — dependency direction is one-way

```
codeSpectra.export, .plotting        ← formats and renders
        ▲
codeSpectra.codes.{asce7, nec, nch433}   ← one subpackage per standard family
        ▲
codeSpectra.core                     ← knows no standard exists
```

`core` must never import from `codes`. `codes` must never import
`matplotlib`. Keeping this straight is what lets a new standard be added
without touching anything else.

| Module | Holds |
|---|---|
| `core.spectrum` | `Spectrum` ABC, `AnalyticSpectrum`, `TabulatedSpectrum`, `SpectrumMeta`, `SpectrumKind` |
| `core.control` | `ControlPeriods` — named breakpoints, and grid refinement |
| `core.tables` | `InterpolatedTable` — the row/column code table with `None` = deferred |
| `core.references` | `ClauseRef` |
| `core.reports` | `Report`, `ReportItem` |
| `core.units` | `AccelUnit`, `from_g`, `STANDARD_GRAVITY` |
| `core.exceptions` | `SiteSpecificRequired`, `InvalidInput`, `TableLookupError` |

---

## 3. `Spectrum` is a value object

Two concretes share one interface, and every operation returns a **new**
spectrum — nothing mutates.

- **`AnalyticSpectrum`** carries a closed-form evaluator, so `at(T)` is exact
  at any period including right on a branch corner. Every branch-defined code
  spectrum uses this.
- **`TabulatedSpectrum`** holds ordinates, interpolates linearly in `T`, and
  applies the ASCE 7-22 §11.4.5.1 decay rule past its last period. Used for
  multi-period spectra, site-specific studies and imported curves.

```python
design.scaled(1.5)                      # MCEr
design.reduced(R=8.0, Ie=1.0)           # inelastic
asce.envelope(nec)                      # point-wise maximum
site_specific.floored_by(code, 0.80)    # the ASCE §21.3 80% floor
design.displacement(2.0)                # Sd = Sa·g·(T/2π)²
design.resampled(periods)               # freeze onto a grid
```

Because these compose, a new code requirement is usually **an operation on a
spectrum, not a new class**. Reach for composition before inheritance.

**`grid()` lands on control periods.** A plain `linspace` chamfers the plateau
corners of a code spectrum. `ControlPeriods.refine_grid` injects each
breakpoint plus points a hair either side, so plotted and exported curves
reproduce the published figure. If you write a new sampler, preserve this.

---

## 4. Units: `Sa` is in g, always

Every standard here defines `Sa` as a fraction of gravity, so g is the one
convention needing no translation at the point of definition. Conversion
happens **only** at the export/plot boundary via `AccelUnit`. Periods are
seconds. Do not introduce a second internal unit system.

(This differs from `apeSteel`'s N-mm-tonne-s base on purpose — mixing the two
would be worse than the inconsistency.)

---

## 5. Citations and reports

Every derived quantity carries a `ClauseRef(standard, edition, clause,
equation, table, figure, description)`. `site.report()` returns a frozen
`Report` that renders to text or Markdown for a calculation package.

Two rules that are easy to break:

- **Report text must survive a `cp1252` console.** Windows terminals default
  to it, and these reports are printed by Ecuadorian and Chilean engineers.
  `η`, `≤`, `≥` and Greek generally are out — write `eta`, `<=`, `>=`.
  Spanish accented capitals (`ÑÁÉÍÓÚ`) and `§` are fine; they exist in cp1252.
  Tests assert this per code — keep them.
- **Non-code provenance leads the notes.** If a value came from anywhere
  other than the standard, the note saying so must be `notes[0]`, before the
  numbers. `NECSEDS2015(provenance_note=...)` exists for this.

---

## 6. Per-edition input contracts differ — this is the most common bug

| Class | You supply | Site coefficients |
|---|---|---|
| `ASCE7_10` | `Ss`, `S1`, site class A–F | `Fa`/`Fv` from Tables 11.4-1/2 (7-10 values) |
| `ASCE7_16` | `Ss`, `S1`, site class A–F | `Fa`/`Fv` from Tables 11.4-1/2 (7-16 values) |
| `ASCE7_22` | `SMS`/`SM1`, **or** 22 MPRS ordinates; site class A–F incl. BC/CD/DE | **none — 7-22 deleted the Fa/Fv tables** |

**ASCE 7-22 §11.4.3 reads `Ss`, `S1`, `SMS`, `SM1` directly from the USGS
Seismic Design Geodatabase for the site class.** There is no `Fa·Ss` path.
Anyone "helpfully" adding one is reintroducing a 7-16 concept the standard
removed. Hence two explicit constructors:

```python
ASCE7_22.from_mprs(periods, sa_mcer, site_class="CD", TL=8.0)   # §11.4.5.1, default
ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD") # §11.4.5.2, Exception 2 only
```

### NEC-SE-DS: the ascending branch is off by default

```
Sa = eta·Z·Fa                    for 0 <= T <= Tc
Sa = eta·Z·Fa·(Tc/T)**r          for T > Tc
Sa = Z·Fa·[1 + (eta-1)·T/T0]     for T <= T0   ← NOT the design spectrum
```

NEC restricts that third line to **dynamic analysis, and within it to modes
other than the fundamental**. The design plateau runs from `T = 0`. Turning it
on by default under-estimates short-period demand, which is why
`elastic_spectrum(include_ascending_branch=True)` is an explicit opt-in.

### NCh433: `R*` is one scalar per direction

`R*(T*, Ro)` depends on `T*`, the period of the mode with the greatest
equivalent translational mass — not a function evaluated at every period.
`design_spectrum(T_star, Ro)` scales the whole curve by that single number.

---

## 7. Transcribing a table from a PDF

Read `references/pdf-extraction.md` before starting — it has the full recipe
and the specific failures this repo has already hit. The headline points:

- The `pdftotext` on this machine is **Xpdf 4.00, not Poppler**: no
  `-bbox-layout`. Use `-table` for tabular pages, `-layout` for prose.
- **Locate the value column by regex at end-of-line, not by the header's
  column offset.** One page of NEC Tabla 19 pads its header differently from
  its data rows; slicing by header offset silently dropped that whole page —
  21 towns including Cuenca.
- **Cell wraps go both ways.** A wrapped name usually sits on the line
  *before* its data row, occasionally after. Attach trailing wraps only to
  fields left empty, or you will absorb the table caption as data.
- **Audit the characters.** The NEC PDF font mis-maps three accented
  capitals (`U+00D0`→`Ñ`, `U+250C`→`Ú`, `U+00CB`→`Ó`). Assert that every
  non-ASCII character in an extracted table is one you expect.
- **Cross-check wide tables against a raw (no `-layout`) extraction.**
  Multi-column standard pages interleave cells and the column order can come
  out scrambled — ASCE 7-16 Table 11.9-1 does exactly this.

`scripts/extract_code_table.py` implements the positional extractor with
these lessons baked in, and prints a validation summary. Use it rather than
rewriting the parser each time.

**Validation is not optional.** Before shipping any extracted table, check:
row count, the set of distinct values (do they match the standard's own
discrete set?), no empty fields, no unconsumed parse leftovers, non-ASCII
audit, and a hand-checked spot list of well-known rows.

---

## 8. Deferred values and escape hatches

`InterpolatedTable` stores `None` for cells the standard defers to a
site-specific study, and raises `SiteSpecificRequired` carrying the clause.
It raises even when a `None` merely *brackets* the requested value —
interpolating toward an undefined cell is not a defined operation.

Escape hatches are always explicit and always named after what they do:

- `allow_site_specific_exception=True` applies a **documented code
  exception** (ASCE 7-16 §11.4.8 Exception 1 substitutes the Site Class C
  `Fa`). It does not apply undocumented ones — Exception 1 rescues `Fa` but
  not `Fv`, and the tests pin that.
- `Fa_override` / `Fv_override` / `Fd_override` … accept a value from a study
  the engineer has actually performed.

When adding a code, prefer this shape over a boolean `strict=False`: the
caller should have to say *which* provision they are invoking.

---

## 9. External data and licensing

The repo bundles code tables (facts from the standard) and one derived
dataset. It deliberately bundles nothing else.

| Source | Licence | What ships |
|---|---|---|
| ASCE 7, NEC-SE-DS, NCh433 tables | transcribed provisions | the values, cited |
| GeoNames (place coordinates) | CC BY 4.0 | derived subset + `GEONAMES_ATTRIBUTION` |
| Palacios et al. 2023 PSHA | **no LICENSE — all rights reserved** | **nothing**; reader only |

The rule: **ship the reader, not the data, unless the licence clearly permits
redistribution.** `ContourHazardMap.from_palacios_2023()` reads a local copy
or fetches on an explicit `download=True`; it never touches the network
implicitly. If a user asks to vendor an unlicensed dataset, say plainly that
citation is not a licence and offer the reader-only shape instead.

---

## 10. The quality gate

All four must be clean before anything is committed:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/codeSpectra
python -c "import ast,pathlib; [ast.parse(f.read_text(encoding='utf-8'), feature_version=(3,10)) for f in pathlib.Path('src').rglob('*.py')]"
```

That last one matters: `pyproject.toml` declares `requires-python >= 3.10`,
and PEP 701 syntax (multi-line expressions or nested same-quotes inside an
f-string) parses fine on the 3.12 interpreter here but is a `SyntaxError` on
3.10. It has already slipped in once.

If a table lives in `tables/*.json`, also confirm it reaches a built wheel —
a `package-data` glob that silently misses is invisible until after install:

```bash
python -m build --wheel -q && python -c "import zipfile,glob; print([n for n in zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist() if n.endswith('.json')])"
```

### Naming conventions

The source mirrors the standards' own notation — `Ss`, `S1`, `SDS`, `Fa`,
`Z`, `I`, `R`, `TL`. Renaming those to satisfy PEP 8 would make the code
harder to check against the printed clause, so the colliding `pep8-naming`
rules are switched **off** in `pyproject.toml` (`N801`, `N802`, `N803`,
`N806`, `N812`, `N815`, `N818`, `E741`) along with `SIM300` (which flips
`actual == pytest.approx(expected)` the wrong way round). Don't re-enable
them; do keep the rest of the ruff selection.

---

## 11. Testing conventions

Tests are the product here as much as the code. The categories that matter:

| Category | What it pins |
|---|---|
| **Table integrity** | every transcribed cell, parametrised against the printed table |
| **Branch shape** | `Sa` at `T=0`, on each plateau, and in each decay branch |
| **Continuity** | `Sa(T-ε) ≈ Sa(T+ε)` at *every* control period, across all zone × soil × region combinations |
| **Code rules** | default-site-class floors, deferred-cell raises, documented exceptions |
| **Encoding** | `report().to_text().encode("cp1252")` per code |
| **Provenance** | non-code sources produce a leading warning note |

Write the assertion so a failure names the provision, not just the number.
When a test fails, first ask whether the *expectation* is wrong — several
times in this repo the code was right and the test encoded a
misunderstanding (e.g. the ASCE §12.8-5 floor legitimately governing over the
§12.8-3 cap at long period).

---

## 12. Adding a new standard

Full walkthrough in `references/adding-a-code.md`. The shape:

1. Extract and **validate** the tables (§7), store as `codes/<family>/tables/*.json`
   or module-level `InterpolatedTable`s.
2. Write a frozen dataclass site class: validated `__post_init__`,
   `cached_property` coefficients, `control_periods`, spectrum builders,
   `report()`.
3. Express the spectrum as a closed-form evaluator returning
   `AnalyticSpectrum`; reuse `two_period_spectrum` if the shape matches
   ASCE's.
4. Cite every clause. Add the deferred-cell raises.
5. Tests in all six categories above.
6. Re-export from the family `__init__` and add lazy access in the top-level
   `__init__.__getattr__`.

---

## 13. Known gaps — say these plainly rather than improvising

- No USGS retrieval; `Ss`/`S1`/MPRS are user-supplied.
- NCh433 Eq. 6-11 (alternative `R*` for wall buildings) is not implemented —
  the PDF extraction was too garbled to be sure of the form.
- ASCE Chapter 21 site response analysis itself; only the §21.3 floor is
  expressible, via `floored_by`.
- NEC §3.1.2 hazard curves for essential/special structures (other return
  periods).
- Tabla 19 coordinate coverage is 88%; `covered_by()` reports the rest.

---

## 14. Evaluation status — what this skill is known NOT to do

Two iterations, twelve completed cells (`evals/iteration-1/RESULTS.md`,
`evals/iteration-2/RESULTS.md`). Eleven scored full marks **with or without
this skill loaded**, at 16-25% higher token cost when loaded.

**Its engineering content is not a capability uplift.** The extraction
playbook (§7), the no-invented-values rule (§1) and the refusal discipline
(§8) were all reproduced unaided across six independent runs on three
standards — including Eurocode 8, which the repo had never touched. Agents
independently found the same PDF row-shifts, the same missing radical in
Eq. (3.6), rendered pages as images, and refused to fabricate deferred
values. Read §1 and §7 as documentation of a shared standard, not as
instruction.

**Its measurable contribution is repo conventions**, and it is consistent:

| Behaviour | with skill | baseline |
|---|---|---|
| Verified new package data reaches a built wheel (§10) | 2/2 | 0/2 |
| Avoided misusing an existing abstraction (§3) | 2/2 | 0/2 |

Those are worth the load. Do not expect more.

**The eval's real yield was six library defects**, every one found by an
agent chasing the problem underneath the request rather than the request
itself — including two silently-wrong design values (`ASCE7_22` reporting
`SDS = 0`, `nec_site_from_hazard` defaulting `region="sierra"`). Treat a
confident request as a prompt to look for the real problem beneath it; that
habit has paid better here than any rule in this document.

### Earlier framing


Iteration 1 (`evals/iteration-1/RESULTS.md`) measured **no difference**
between having this skill loaded and not: 20/20 assertions passed in both
configurations, at ~16% higher token cost. The cause is worth knowing:

- For the two trap cases, the answers are already in the module docstrings of
  `asce7_22.py` and `nec_se_ds_2015.py`. **Where the source documents a
  provision, this skill is redundant** — read the source first.
- For the extraction case, both configurations independently cross-checked
  render modes and refused to fabricate a missing value. The extraction
  discipline in §7 is worth stating, but do not assume it is load-bearing.

The one thing only the skilled run did was verify a claim against the primary
source rather than asserting it from a docstring. If that is all you need,
§7's recipe is the part to read.

The eval also surfaced two real library defects (a silently-zero `SDS` on
`ASCE7_22`, since fixed, and dead dataclass fields) — both found by the
*unskilled* run. Treat a confident request as a prompt to look for the real
problem underneath it.

---

## 15. Environment gotchas

- **`isolation: worktree` seeds from repo HEAD, not the working branch.** An
  agent given a worktree may land on `main`, which predates the library
  rewrite. Check `git log --oneline -1` before trusting the tree.
- **The editable install resolves `codeSpectra` to one specific worktree.** A
  bare `pytest` from a sibling worktree silently tests the wrong checkout —
  it has already produced a passing run against code the agent had not
  written. Run with `PYTHONPATH=<your worktree>/src`.
- **Doctests do not run in the default suite.** `pytest --doctest-modules`
  catches docstring examples that drifted.
- **Whole-repo `mypy` reports pre-existing errors in `tests/`.** The gate in
  §10 checks `src/codeSpectra` only; that is deliberate.

---

## 16. Cross-references

- **Ground-motion theory, site classes, MPRS vs two-period** —
  `asce7-ground-motion`.
- **ELF, drift, demands** — `asce7-seismic-demands`; **SDC and system
  selection** — `asce-seismic`.
- **Standards PDFs on this machine** —
  `C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos`, by issuing
  body (`ASCE/`, `Ecuador/NEC/`, `Chile/`).
- **Downstream FEM** — `apegmsh-helper`; **steel member checks** —
  `apesteel-helper`.
