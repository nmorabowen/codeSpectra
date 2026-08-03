Eurocode 8 is in. New `codeSpectra.codes.en1998` subpackage with the EN 1998-1
§3.2.2.2 Type 1 horizontal elastic response spectrum, ground types A–E, and the
Eq. 3.6 damping correction.

```python
from codeSpectra import EN1998_1

site = EN1998_1(ag=0.25, ground_type="C")     # or agR=0.20, gamma_I=1.25
site.S, site.TB, site.TC, site.TD             # 1.15, 0.20, 0.6, 2.0 — Table 3.2

five  = site.elastic_spectrum()               # eta = 1.000
two   = site.elastic_spectrum(damping=2.0)    # eta = 1.195
seven = site.elastic_spectrum(damping=7.0)    # eta = 0.913
```

`damping` is ξ **as a percentage**, feeding `eta = sqrt(10/(5 + xi)) >= 0.55`.

## What's in it

- **Table 3.2** — `S`, `TB`, `TC`, `TD` for ground types A through E.
- **Eqs. 3.2–3.5**, the four branches, as an `AnalyticSpectrum` so `at(T)` is
  exact right on `TB`/`TC`/`TD` and `grid()`/`export` reproduce the corners of
  Figure 3.2 instead of chamfering them.
- **Eq. 3.6** damping correction, with the 0.55 floor. Exposed both as
  `site.eta(xi)` and as a standalone `damping_correction(xi)`.
- `ag` in **g** (the library's one unit), or `agR` + `gamma_I` per §3.2.1(3),
  with a cross-check that rejects an inconsistent pair.
- `report()` with a `ClauseRef` on every line, cp1252-safe for a Windows
  console.

One thing worth knowing about the damping factor: `eta` scales Eqs. 3.3–3.5
outright, but in Eq. 3.2 it only multiplies the *slope* term. So
`Se(0) = ag·S` is identical at 2%, 5% and 7% — the PGA anchor isn't damped,
only the amplified part of the curve is. There's a test pinning that, because
it's an easy thing to "fix" wrongly.

## Three deliberate refusals

**`S`, `TB`, `TC`, `TD` are Nationally Determined Parameters.** Table 3.2 is
only the *recommended* set; §3.2.2.2(2)P NOTE 1 hands the real values to each
country's National Annex. That's the single biggest way this could give a
Spanish or Italian engineer a wrong number, so the report always says so, and
`S_override` / `TB_override` / `TC_override` / `TD_override` take Annex values.
Use one and the report leads with a note that the value did *not* come from
Table 3.2. No National Annex values are bundled — I have no source for them.

**Ground types S1 and S2 raise `SiteSpecificRequired`,** not a plausible
number. §3.1.2(4)P and NOTE 2 to §3.2.2.2(2)P require a special study; Table
3.2 tabulates nothing for them. The exception carries both clauses and points
at the overrides as the way through once the study exists.

**The §3.2.2.5 design spectrum `Sd(T)` is not implemented, and
`.reduced(q)` is not a substitute for it.** `Sd(T)` is a genuinely different
shape (`2/3 + T/TB·(2.5/q − 2/3)` on the first branch, plus the `beta·ag` lower
bound), not `Se(T)/q`. Also absent: the Type 2 spectrum (Table 3.3), the
vertical spectrum §3.2.2.3, and the Annex A displacement spectrum past 4 s.
The `report()` notes and the README say all of this out loud.

## Verification

I read your PDF rather than working from memory, and Table 3.2 needed it —
`pdftotext -layout` renders that page with the ground-type column offset from
the period columns, so a careless read pairs ground type A with C's periods.
Cross-checked against `-table` and `-raw` (which agree row-for-row) and then
against a 200 dpi render of the page. Same for Eq. 3.6: the text extraction
drops the radical entirely and reads `η = 10/(5+ξ) ≥ 0,55`, which is a real
trap because both forms give η = 1 at 5% damping. The rendered page shows the
square root. Final check: the plateau ordinates come out at 2.5, 3.0, 2.875,
3.375, 3.5 × ag for A–E, matching Figure 3.2.

Because the source is `prEN 1998-1:2003`, the December 2003 final draft, every
report carries a note saying so — worth confirming against the published
EN 1998-1:2004 as adopted where you're building.

## Tests and gate

117 new tests in `tests/test_en1998.py`, covering all six categories the repo
uses: every Table 3.2 cell against the printed table, each branch equation
hand-computed, continuity at `TB`/`TC`/`TD` across all five ground types × five
damping values, the S1/S2 raise and the `eta` floor, cp1252 report encoding,
and the National-Annex provenance note leading. 629 pass overall; ruff, mypy
and the 3.10 AST check are clean.

README and the standards table are updated. Nothing committed.
