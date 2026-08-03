Eurocode 8 is in. `EN1998_1` implements §3.2.2.2 — the horizontal elastic
response spectrum, Eq. (3.2)–(3.5), for ground types A through E, with the
Eq. (3.6) damping correction wired through every branch.

```python
from codeSpectra import EN1998_1

site = EN1998_1(a_gR=0.25, ground_type="C")   # Type 1 by default
site.S, site.TB, site.TC, site.TD             # 1.15, 0.20, 0.6, 2.0 — Table 3.2

site.elastic_spectrum()                       # 5% damped, eta = 1
site.elastic_spectrum(damping=0.02)           # eta = 1.1952
site.elastic_spectrum(damping=0.07)           # eta = 0.9129
site.eta(0.02)                                # the factor on its own
```

It behaves like every other spectrum in the library — `at()`, `grid()`,
`reduced()`, `envelope()`, `displacement()`, `plot()`, the exporters, and a
cited `report()`.

## Three things worth knowing

**Damping is a fraction, not a percentage.** Eq. (3.6) is written as
`η = √(10/(5+ξ)) ≥ 0.55` with `ξ` in percent, but every other damping value in
codeSpectra is a fraction. So the API takes `0.02`, converts internally, and
**raises** if you pass `2` or `5`. Silently accepting `5` would return
`η = 1.0` for a 500%-damped structure — right answer, wrong reason, and
invisible. The 0.55 floor is implemented; it engages just past ξ = 28%.

**`η` scales every branch except the T = 0 intercept.** From Eq. (3.2),
`Se(0) = ag·S` regardless of damping — the `η` term is multiplied by `T/TB`,
which vanishes at the origin. So a 2%-damped curve and a 5%-damped curve start
at the same point and diverge from there. There is a test for exactly this,
because it is the easy thing to get wrong when factoring `η` out front.

**Tables 3.2 and 3.3 are recommendations, not requirements.** They live in the
*informative* NOTE 1 to §3.2.2.2(2)P; the National Annex of the country of use
governs. So `S`, `TB`, `TC`, `TD` are each individually overridable, the report
carries a standing note about it, and it flags loudly when an override is in
play:

```python
EN1998_1(a_gR=0.25, ground_type="C", S_override=1.25, TC_override=0.5)
```

You are working in Europe across several National Annexes — that hatch is the
part you will actually use. I did not bundle any NA parameter sets; guessing at
27 of them is exactly the kind of invented data this library refuses to ship.

## Also included

I added the **Type 2** shape (Table 3.3) alongside Type 1. It is the same
clause, the same equations, and the recommended shape wherever `Ms ≤ 5.5`
dominates the hazard — a lot of central and northern Europe. Type 1 is the
default; `spectrum_type=2` switches. Both tables were transcribed and verified
in the same pass, so it cost nothing to get right.

`ag = γ_I · a_gR` per §3.2.1(3), so `gamma_I=` is a constructor argument rather
than something you pre-multiply.

Ground types **S1 and S2** raise `SiteSpecificRequired` — Note 2 to
§3.2.2.2(2)P tabulates nothing for them. If you have done the study, supply all
four of `S_override`/`TB_override`/`TC_override`/`TD_override` and it proceeds.

## Verification

Every number came out of your PDF, not memory. `pdftotext -layout` mangles
Table 3.2 — it shifts the `S` column one line down relative to `TB`/`TC`/`TD`,
so a straight read pairs ground type C with B's periods. I rendered pages 24–26
to images and read the tables and Eq. (3.6) off the rendering to confirm the
alignment; the radical sign in Eq. (3.6) is dropped entirely by text extraction,
which would have left `η = 10/(5+ξ)`.

As an independent check, the implementation reproduces Figure 3.2 and Figure
3.3 exactly: Type 1 plateaus at `Se/ag` = 2.5, 3.0, 2.875, 3.375, 3.5 for A–E;
Type 2 at 2.5, 3.375, 3.75, 4.5, 4.0 — matching the curve ordering annotated on
both figures.

141 new tests. Every cell of Tables 3.2 and 3.3 asserted, each of Eq. (3.2)–
(3.6) checked against its own algebra, continuity verified at `TB`, `TC`, `TD`
across all 5 ground types × 2 shapes × 3 damping levels, plus the validation and
override behaviour. Full suite 653 passed; `ruff` clean; `mypy --strict` adds no
new errors (the 10 that remain are pre-existing, all in `tests/test_export.py`,
`tests/test_core.py` and `tests/test_nec_hazard.py`).

## Not implemented

Deliberately scoped to §3.2.2.2:

- §3.2.2.3 vertical spectrum (`avg`, factor 3.0, Table 3.4)
- §3.2.2.5 design spectrum for elastic analysis — the behaviour factor `q` and
  the `β·ag` lower bound. Say the word; it is the obvious next piece, and it is
  what you will need for actual member design rather than just demand curves.
- §3.2.2.4 design ground displacement `dg`
- Annex A, which extends the acceleration spectrum past 4 s. Above 4 s the
  implementation keeps applying Eq. (3.5), which the clause does not sanction —
  `t_max` defaults to 4 s so sampling, plotting and export stop there, and the
  report says so.
