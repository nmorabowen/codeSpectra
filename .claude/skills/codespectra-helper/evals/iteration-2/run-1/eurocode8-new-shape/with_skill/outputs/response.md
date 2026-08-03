Eurocode 8 is in. `EN1998_1` builds the §3.2.2.2 Type 1 horizontal elastic
spectrum for ground types A–E, with the Eq. (3.6) damping correction.

```python
from codeSpectra import EN1998_1

site = EN1998_1(agR=0.25, ground_type="C", importance_class="II")
site.ag, site.S, site.TB, site.TC, site.TD   # 0.25, 1.15, 0.20, 0.60, 2.0

elastic = site.elastic_spectrum()            # 5% damped
elastic.at(0.0)                              # 0.2875 g  = ag·S
elastic.at(0.4)                              # 0.71875 g = ag·S·η·2.5

site.with_damping(0.02).elastic_spectrum()   # η = 1.1952 — curve up 19.5%
site.with_damping(0.07).elastic_spectrum()   # η = 0.9129 — curve down 8.7%
```

All four branches, Eqs. (3.2)–(3.5), plus `ag = γI·agR` from §3.2.1(3).

## The one thing worth flagging

**`pdftotext` silently drops the square root in Eq. (3.6).** Every text render
mode of your PDF gives

```
η = 10 /(5 + ξ ) ≥ 0,55        (3.6)
```

The actual expression is `η = √(10/(5+ξ)) ≥ 0,55` — I confirmed it against a
200-dpi image of page 26. Both forms give η = 1 at 5% damping, so the obvious
sanity check does not catch it, but the un-rooted version over-predicts a
2%-damped spectrum by 20% and puts the 0,55 floor at 13% damping instead of
28%. `test_square_root_is_not_dropped` pins it so it cannot regress.

## Design choices you may want to overrule

- **`damping` is a field on the site object, not an argument to
  `elastic_spectrum()`.** That way `report()` and the spectrum can never
  disagree about which η was used. `with_damping()` returns a new site;
  nothing mutates.
- **Table 3.2 values are the *recommended* ones.** §3.2.2.2(2)P NOTE 1 leaves
  `S`, `TB`, `TC`, `TD` to the National Annex, and §4.2.5 does the same for
  `γI`. Pass `S_override` / `TB_override` / `TC_override` / `TD_override` /
  `gamma_I_override` for your NA; the report flags the caveat either way.
- **Ground types S1 and S2 raise `SiteSpecificRequired`** rather than
  returning a plausible number — NOTE 2 to §3.2.2.2(2)P defers them to a
  special study. Supply all four parameters and they proceed. Supplying only
  some raises and names the missing ones.
- **`t_max = 4.0 s`**, because that is the range Eq. (3.5) is written for.
  §3.2.2.2(6) sends you to Informative Annex A beyond that.

The elastic displacement spectrum, Eq. (3.7) `SDe = Se·(T/2π)²`, is already the
generic `spectrum.displacement(T)` — no new code, and a test asserts they agree.

## Deliberately not implemented

Say the word and any of these is a small addition, but I would rather not ship
values I have not read off the page:

- **Table 3.3, the Type 2 spectrum.** You will hit this: §3.2.2.2(2)P NOTE 1
  recommends it wherever the dominant earthquakes have `Ms ≤ 5,5`, which is
  much of central and northern Europe. I validated Table 3.3 in the same
  extraction, so wiring it in is a table plus a `spectrum_type` argument.
- §3.2.2.3 vertical spectrum (Eqs. 3.8–3.11, Table 3.4).
- §3.2.2.5 design spectrum for linear analysis — the behaviour factor `q` and
  the `β·ag` lower bound. Note this is *not* `spectrum.reduced(q)`: §3.2.2.5
  has a different low-period branch and a floor, so I did not want anyone
  assuming the generic reduction stands in for it.

## Provenance

Your PDF is **prEN 1998-1:2003 (E)**, the December 2003 draft, not the
published EN 1998-1:2004. §3.2.2.2 and Table 3.2 did not change between them
as far as I know, but I only verified the draft, so that is what the citations
and the leading report note say. Worth a check against the final text before
this goes into a real calculation package.

Table 3.2 was cross-checked in three independent render modes plus the page
image — the `-layout` mode scrambled it by one row (it reads C as `1,15` with
A's periods), which is exactly the column-desync failure the extraction
playbook warns about.

## Quality gate

`pytest` (611 passed, 99 of them new), `ruff`, `mypy --strict` on `src/codeSpectra`,
and the 3.10 AST parse are all clean; doctests pass; the new subpackage
reaches a built wheel.
