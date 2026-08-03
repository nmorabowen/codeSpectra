Eurocode 8 is in. New package `codeSpectra.codes.ec8`, exported at the root as `EN1998_1`.

```python
from codeSpectra import EN1998_1

site = EN1998_1(agR=0.25, ground="C", importance_class="III")
site.S, site.TB, site.TC, site.TD   # (1.15, 0.2, 0.6, 2.0)
site.ag                             # 0.30 g = gamma_I * agR
se = site.elastic_spectrum()        # Se(T) in g, reported to 4 s

EN1998_1(agR=0.25, ground="C", damping=2.0).eta   # 1.1952
EN1998_1(agR=0.25, ground="C", damping=7.0).eta   # 0.9129
```

The result is an ordinary `AnalyticSpectrum`, so `reduced()`, `envelope()`, `displacement()`, `to_etabs()` and the rest all work on it unchanged.

## What it implements

§3.2.2.2 Eqs. (3.2)–(3.5), the Type 1 four-branch shape, with Table 3.2 ground types A–E, plus Eq. (3.6) `η = sqrt(10/(5+ξ)) ≥ 0.55`. `ag = γI·agR` per §3.2.1(3), with γI either from an importance class (Table 4.3) or given directly.

## Three things I want to flag

**`ξ` goes in as a percentage.** Eq. (3.6) is written in percent, so `damping=2.0` means 2%. Passing `0.05` expecting 5% would silently give `η = 1.407` — a 41% overestimate — so any value in `(0, 1)` is rejected with a message pointing at the mix-up. If you genuinely want sub-1% damping, pass the parameters through the overrides instead.

**`η` does not simply scale the curve.** It multiplies the plateau and both decaying branches, but on the ascending branch it sits *inside* the bracket of Eq. (3.2). So `Se(0) = ag·S` at every damping ratio — the zero-period ordinate is the ground acceleration and cannot depend on structural damping. Below `TB` the 2%-damped curve is only partly above the 5% one. There is a test asserting the ratio at `T = 0.1` s lies strictly between 1 and `η`, because "multiply the whole spectrum by η" is the natural wrong implementation.

**Everything in Table 3.2 is a Nationally Determined Parameter.** 3.2.2.2(2)P NOTE 1 makes `S`, `TB`, `TC`, `TD` National Annex values, and several member states do depart from the recommended set. Same for γI outside class II. The values shipped are the recommended ones and every report says so. `S_override` / `TB_override` / `TC_override` / `TD_override` are the hook for your National Annex — and for ground types `S1` and `S2`, which have no tabulated parameters at all and raise `SiteSpecificRequired` until you supply the special-study values.

## A transcription trap worth knowing about

Table 3.2's text layer in that PDF interleaves the ground-type letters with the wrong numeric rows. Read naively it yields `S` = 1.0, 1.2, 1.15, 1.35, 1.4 paired against `TB`/`TC` triples shifted by one row — plausible-looking and wrong. I rendered printed page 24 as an image and read the table off the rendered page instead, then cross-checked the resulting curves against Figure 3.2: plateaus land at 2.5 / 3.0 / 2.875 / 3.375 / 3.5 × `ag` and the plateaus end at `TC` = 0.4 / 0.5 / 0.6 / 0.8 / 0.5 s, matching the published figure. `test_soil_factor_is_not_monotonic_in_ground_type` asserts `S(C) < S(B)` specifically to catch a future row-shifted re-transcription.

## Not implemented (deliberately, and noted in every report)

Type 2 (Table 3.3, recommended where `Ms ≤ 5.5` dominates the hazard), the §3.2.2.3 vertical spectrum, the §3.2.2.5 design spectrum with behaviour factor `q`, and Informative Annex A beyond 4 s. Say the word on any of them — Type 2 in particular is a table and a flag, since the shape function is identical.

One caveat on provenance: the PDF you pointed me at is `prEN 1998-1:2003`, the December 2003 final draft, so that is what the citations say and what the reports name as the transcription source. If you have the published EN 1998-1:2004 to hand, worth a spot-check of Table 3.2 before this goes on a real project.

## Tests

98 new tests in `tests/test_ec8.py`, covering every Table 3.2 cell, each of the four branches against its own equation, continuity at `TB`/`TC`/`TD`, the `η` curve including its 0.55 floor and the percent-vs-fraction guard, the importance factors, the `S1`/`S2` site-specific trigger, and the override paths. Full suite: 610 passed. `ruff` clean, `mypy --strict` clean, doctests pass.
