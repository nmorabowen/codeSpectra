# Iteration 2 — results

Run 2026-08-03 against skill commit `198bbad`. Two cases, two runs per
configuration (8 cells), chosen to test what iteration 1 could not: knowledge
the source does **not** already carry.

## Headline

| | Pass rate | Tokens | Time |
|---|---|---|---|
| with skill | **1.000** ± 0.000 | 202,722 | 1,375 s |
| baseline | **0.938** ± 0.108 | 161,937 | 850 s |
| **delta** | **+0.062** | +40,786 (+25%) | +526 s (+62%) |

| Case | with skill | baseline |
|---|---|---|
| `eurocode8-new-shape` | 11/11, 11/11 | 11/11, 11/11 |
| `osm-licence-boundaries` | 8/8, 8/8 | **6/8**, 8/8 |

The entire measured delta is one baseline cell dropping two assertions. That
is a single sample, and the baseline standard deviation (0.108) is larger than
the delta (0.062). **Treat this as a tie with one interesting outlier, not as
a demonstration that the skill works.**

## Eurocode 8 — a clean tie, and a strong one

All four runs independently:

- found that `pdftotext -layout` **row-shifts Table 3.2**, pairing ground
  types with the wrong `TB`/`TC`/`TD`;
- found that **the radical is missing from Eq. (3.6)** in the text layer,
  yielding `η = 10/(5+ξ)` instead of `√(10/(5+ξ))` — dangerous because both
  forms give `η = 1` at the 5% reference damping, so no sanity check catches
  it. The un-rooted form over-predicts a 2%-damped spectrum by ~19.5%;
- fell back to rendering pages as images;
- cross-checked plateau ratios (2.5 / 3.0 / 2.875 / 3.375 / 3.5 × `ag`)
  against published Figure 3.2;
- flagged Table 3.2 as Nationally Determined Parameters and made ground types
  S1/S2 raise rather than invent values;
- cited `prEN 1998-1:2003` — the draft — rather than overclaiming the
  published 2004 edition.

The extraction playbook in §7 is therefore **not load-bearing for this model**.
That was already suspected after iteration 1; it is now established across
four independent runs on a standard the repo had never touched.

### Three behaviours that did split

| Behaviour | with skill | baseline |
|---|---|---|
| Verified the new package reaches a built wheel (§10) | 2/2 | 0/2 |
| Warned §3.2.2.5 `Sd(T)` is *not* `Se(T)/q` — don't use `.reduced()` | 2/2 | 0/2 |
| Noticed Eq. (3.7) needs no new code — it is `Spectrum.displacement()` | 1/2 | 0/2 |

These are **library-integration** behaviours, not engineering ones: the
packaging gate from §10 and §3's "a new code requirement is usually an
operation on a spectrum". None were assertions, so none show in the score.
This is the clearest signal the skill produced across both iterations, and it
is worth exactly what it looks like: small, real, and about the repo's own
conventions rather than about seismic engineering.

### A variance finding n=1 would have hidden

The two baseline runs made **opposite API decisions on the same ambiguity**:

| | `ξ` convention | Guard |
|---|---|---|
| run 1 | percentage (`5.0` = 5%) | rejects values in `(0,1)` |
| run 2 | fraction (`0.05` = 5%) | rejects values `≥ 1` |

Both identified the ambiguity as dangerous (`η` is off by ~41% if confused)
and both guarded it — in opposite directions. Code written against one raises
against the other. If EC8 is ever added for real, the convention belongs in
§6's per-code contracts table.

## The licence case — the one real split, and not where expected

All four runs reached the same legal analysis unaided: OSM is ODbL 1.0, a
boundary extract is a Derivative Database, share-alike attaches, and this is
categorically unlike the attribution-only CC BY GeoNames bundle already
shipped.

They split on the **engineering** call:

| | Decision |
|---|---|
| with skill, run 1 | bundled, with `LICENSE` third-party section, a `tables/LICENSE-OpenStreetMap.txt`, and `OSM_ATTRIBUTION` in every derived-region note |
| with skill, run 2 | bundled, having first *measured* that the no-new-data route was insufficient |
| baseline, run 1 | bundled, with ODbL notice and attribution constant |
| baseline, run 2 | refused; found the data was not needed at all |

The one cell that dropped assertions (baseline run 1, 6/8) bundled without
surfacing that it changes the distributed package's terms, and without
evaluating an alternative source. Every other cell did one or the other.

Two pieces of reasoning stand out, neither attributable to the skill:

- **with skill, run 2** quantified the alternative before spending licence
  budget: the existing Tabla 19 + gazetteer route disagrees with true
  point-in-polygon on **6.3% of Ecuador's land area**, worst case reading a
  Sierra site as Costa (`eta` 2.48 → 1.80, a 27% plateau under-estimate). It
  then evaluated the CC0 alternative (geoBoundaries) and rejected it with
  specifics — Cotopaxi and Chimborazo both tagged `EC-H`, provenance a bare
  Wikimedia URL with `licenseDetail: nan`, 2011 vintage.
- **baseline, run 2** went the other way and found the data unnecessary,
  because Tabla 19 already names the provincia of all 515 poblaciones.

Both are good answers. The skill did not determine which was reached.

## The eval design was flawed again, and I said otherwise

I claimed this case tested skill-only knowledge because `ODbL`,
`share-alike` and `OpenStreetMap` appear nowhere in `src/`. That was checking
for the wrong strings. The **no-bundling stance** is in the source in three
places — `hazard.py`'s module docstring, `PALACIOS_2023.licence`, and a test
named `test_no_data_is_bundled` — and a baseline agent cited all three by
name. Only the ODbL fact itself is absent, and that is general world
knowledge, not something a skill supplies.

Separately, the assertion "does not bundle OSM-derived data" was **stricter
than the skill's own rule**. §9 says ship the reader "unless the licence
clearly permits redistribution", and ODbL *does* permit redistribution. The
assertion encoded a preference, not the documented rule, and was replaced
mid-grading with assertions about whether the obligations were identified and
the change of terms surfaced.

## What the run produced that the score does not

Three real defects in shipped library code, all found by subagents, none
asked about by any assertion:

1. **`nec_site_from_hazard` silently defaulted `region="sierra"`.** An omitted
   argument gave a Guayaquil site `eta = 2.48` instead of 1.80 — a 38% higher
   plateau — and an Oriente site 2.48 instead of 2.60, understating it. A PGA
   contour map carries no province, so the function cannot infer region.
   `region` is now required.
2. **`region_for_provincia("Santo Domingo de los Tsáchilas")`** returned
   "Unknown Ecuadorian province" instead of the §3.3.1 explanation, because
   Tabla 19 abbreviates it `STO.` and only that spelling matched the key.
   Fixed by expanding abbreviations on the lookup path.
3. **Canton La Concordia** transferred Esmeraldas → Santo Domingo de los
   Tsáchilas in 2013, after NEC-SE-DS. Tabla 19 still lists `LA CONCORDIA`
   and `PLAN PILOTO` under Esmeraldas, and the outcomes diverge — Esmeraldas
   gives the Sierra `eta`, Santo Domingo has no assignment and raises. The
   table is reproduced as printed (the code governs); the conflict is now
   documented and pinned by tests.

Also surfaced: the `package-data` glob needed widening to `*.txt` or a
compliance notice would have been dropped from every install; and an Overpass
mirror returned HTTP 200 with zero elements for 16 of 24 provinces — valid
JSON, no data — caught only because the agent validated the response.

## Conclusion across both iterations

Eleven of twelve completed cells scored full marks with or without the skill.
The skill's measurable contribution is confined to **repo conventions**: the
wheel-packaging gate, and knowing which existing abstraction to reuse or not
misuse. Its engineering content — the extraction playbook, the
no-invented-values rule, the refusal discipline — is knowledge this model
already brings, demonstrated across six independent runs on three standards.

That is worth keeping §10 and §3 for, and worth being honest that §1 and §7
are documentation of a shared standard rather than a capability uplift.

The exercise's real return has been **six defects found in the library** —
two in iteration 1, three here, plus the packaging glob — every one of them
by an agent chasing the problem underneath a request rather than the request
itself.

## If there is an iteration 3

Do not write another eval whose answer the codebase already documents. The
untested surface now is:

- **behaviour under a wrong premise stated with authority** — every case so
  far let the agent verify against a PDF; try one where the source of truth
  is absent and the honest answer is "I cannot confirm this";
- **cross-code work** — enveloping ASCE against NEC, where the trap is unit
  and hazard-level mismatch rather than a table value;
- **the export formats** — nothing has tested whether an ETABS or OpenSees
  file is actually ingestible.

And keep two runs per cell minimum: the only two genuine divergences found in
this entire exercise (the `ξ` convention, the bundling decision) were both
within-configuration, and n=1 would have shown neither.
