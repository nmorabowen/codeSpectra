I didn't add the tables, because ASCE 7-22 doesn't have any. This isn't an oversight in `asce7_22.py` — it's the module faithfully tracking a deliberate change in the standard.

## Why there are no Fa/Fv tables to add

Eliminating the site coefficients was one of the headline changes in 7-22's seismic ground motion provisions. Tables 11.4-1 and 11.4-2 were deleted outright. In their place, §11.4.3 has you read `SMS` and `SM1` **directly from the USGS Seismic Design Geodatabase for your site class** — the site amplification is already baked into the multi-period response spectra the geodatabase returns, so there's nothing left for an `Fa` to do.

So `SMS = Fa·Ss` and `SM1 = Fv·S1` are 7-10 and 7-16 relationships. In 7-22 they simply don't exist, and there is no code-sanctioned arithmetic that gets you from mapped `Ss`/`S1` plus a site class to `SMS`/`SM1`. If I'd written Fa/Fv tables into `asce7_22.py`, I'd have been inventing numbers and stamping a 7-22 citation on them — the tables would either be 7-16's values wearing the wrong label, or made up.

Two further things make the borrow genuinely incoherent rather than merely unofficial:

- 7-22 added Site Classes **BC, CD and DE**, and the default site class is now the most critical of C/CD/D. Three of the eight classes have no 7-16 coefficient at all, and the default case has no 7-16 analogue.
- 7-22's `Ss`/`S1` values themselves come from new ground motion models on a multi-period basis. Even for Site Class D, 7-16 `Fa` applied to 7-22 `Ss` is mixing two different hazard characterizations.

## What I did fix

Your complaint was pointing at something real, though — just one layer down. The class was letting the 7-16-shaped call succeed and then quietly produce nonsense:

```python
>>> s = ASCE7_22(Ss=1.5, S1=0.6, site_class='D')   # constructed fine
>>> s.SDS, s.SD1
(0.0, 0.0)                                          # silently zero
>>> s.design_spectrum()
InvalidInput: A tabulated spectrum needs at least two points.
```

`Ss` and `S1` are optional metadata fields on `ASCE7_22`, so that call bound them happily, left `SMS`/`SM1` at their `0.0` defaults, and handed back a zero spectrum. The only complaint came much later, from deep inside the spectrum builder, and said nothing about the actual problem. That's a bad failure mode for a design-code library — a zero `SDS` is the kind of thing that can survive into a model.

It now fails immediately, at construction, and explains itself:

```
ASCE7_22 was given Ss/S1 but no site-adjusted values. On this class Ss and S1
are reporting metadata only and drive no calculation. ASCE 7-22 has no Fa/Fv
site coefficients: §11.4.3 reads SMS and SM1 directly from the USGS Seismic
Design Geodatabase for the applicable site class, so mapped Ss/S1 plus a site
class cannot be converted to SMS/SM1 the way ASCE 7-10 and 7-16 allow. Build
with ASCE7_22.from_mprs(periods, sa_mcer, ...) from the geodatabase
multi-period MCEr ordinates (§11.4.5.1, the default basis), or with
ASCE7_22.from_site_adjusted(SMS=..., SM1=...) where §11.4.5 Exception 2
applies. [...]
```

I also made `Ss`/`S1` earn their keep when you *do* pass them alongside `SMS`/`SM1` — they now appear as report items labelled "reported only; no Fa applies", so the report shows what you started from without implying a conversion happened.

## What to do with the Ss and S1 you have

**If you're doing real 7-22 work:** the same USGS query that gives you `Ss`/`S1` also returns `SMS`, `SM1` and the 22 multi-period ordinates for your site class. Grab the MPRS and use the default path — it's what §11.4.5.1 wants anyway, and the two-period path is only permitted under §11.4.5 Exception 2 where multi-period values aren't available:

```python
site = ASCE7_22.from_mprs(periods, sa_mcer, site_class="CD", TL=8.0)
```

**If you want a preliminary number before that query,** borrow the 7-16 coefficients explicitly, so the provenance is visible in your own code rather than hidden behind a constructor:

```python
s16  = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
site = ASCE7_22.from_site_adjusted(SMS=s16.SMS, SM1=s16.SM1,
                                   site_class="D", TL=8.0)
```

I documented this in the README and covered it with a test, but deliberately did **not** wrap it in a classmethod. As a one-liner in your script it reads as "I am knowingly using 7-16 coefficients"; as `ASCE7_22.from_ss_s1(...)` it would read as a 7-22 provision, which is exactly the confusion that prompted your message.

If the ergonomics still bite, the thing worth building is a geodatabase client — `ASCE7_22.from_usgs(lat, lon, site_class, risk_category)` hitting the USGS Design Maps web service and returning the MPRS. That would make 7-22 *easier* to use than 7-16 rather than harder, and it would be code-correct. Say the word and I'll take a look at it.

**Changed:** `src/codeSpectra/codes/asce7/asce7_22.py`, `tests/test_asce7.py` (5 new tests), `README.md`. Full suite passes (509 tests).

One environment note: `codeSpectra` is pip-installed in editable mode pointing at your `asce-nec-spectrum-library-5739fa` worktree, so a bare `pytest` in any other worktree silently tests *that* checkout instead of the one you're editing. I worked around it with `PYTHONPATH`, but it's worth knowing about — it will bite you again.
