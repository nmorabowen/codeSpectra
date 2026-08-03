I didn't add the tables, because ASCE 7-22 doesn't have them to transcribe. The inconsistency you're seeing is real, but it's in the standard, not in the library.

## What 7-22 actually did

ASCE 7-22 §11.4.3, in full:

> Risk-targeted maximum considered earthquake (MCE_R) spectral response acceleration parameters S_S, S_1, S_MS, and S_M1 shall be obtained from the USGS Seismic Design Geodatabase for the applicable site class.

All four come out of the geodatabase, already site-adjusted. There is no `S_MS = Fa*S_S` equation in 7-22 and no Fa/Fv table behind it. I checked the PDF: the strings `Table 11.4-1` and `Table 11.4-2` occur **zero** times in the whole standard, commentary included. The 7-16 §11.4.4 "Site Coefficients" subsection is gone from the section list entirely — 7-22 goes §11.4.3 (MCE_R parameters) straight to §11.4.4 (Design spectral acceleration parameters, Eqs. 11.4-1/11.4-2, the 2/3 factor).

Commentary C11.4.3 says why in as many words:

> Seismic parameters S_MS and S_M1 (and S_DS and S_D1) incorporate site effects, eliminating the need for the tables of site factors Fa and Fv of ASCE 7-16.

That's the whole point of the Project '17 change. The 7-16 Fa/Fv tables were coarse, period-independent multipliers on a Site Class B/BC reference motion; 7-22 replaced them with site-class-specific hazard computed at 22 periods. So if I put the 7-16 Fa/Fv values into `asce7_22.py`, `Fa * Ss` would give you a number that is neither a 7-16 result (wrong mapped values, wrong reference site condition) nor a 7-22 result (7-22 has no such operation). Nobody could check it against a code book, which is the one thing this library is for.

## What to do with the S_s and S_1 you have

It depends where they came from:

**You have a latitude/longitude.** Then you don't actually need S_s/S_1. Ask the [ASCE 7 Hazard Tool](https://asce7hazardtool.online/) (or the USGS Seismic Design Web Service) for the site class you're designing to and it hands back S_MS, S_M1 *and* the 22 multi-period MCE_R ordinates for that same point — same query, same effort. Then:

```python
# Preferred: §11.4.5.1, the default path in 7-22
site = ASCE7_22.from_mprs(MPRS_PERIODS, sa_mcer, site_class="CD", TL=8.0)

# Or, if the tool has no multi-period values for your point (§11.4.5 Exception 2)
site = ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD")
```

**You're reading the printed Chapter 22 maps.** Those plot S_MS and S_M1 directly in 7-22, for the default site conditions of §11.4.2.1 (the envelope of Site Classes C, CD and D). They are not S_s/S_1 awaiting a coefficient. Read them off and pass them to `from_site_adjusted(..., default_site_conditions=True)` — the report will then remind you to confirm the envelope applies.

**Your S_s/S_1 came off a 7-16 or IBC job.** Then they're 7-16 mapped values on a different reference site condition, and the right object for them is `ASCE7_16`, which does own the 7-16 tables. Mixing them into a 7-22 spectrum isn't a shortcut, it's a third edition that doesn't exist.

## What I did change

Three small things, all cited, no new numbers:

1. **`asce7_22.py` module docstring** — spelled out the C11.4.3 rationale, and added the routing above so the next person who arrives holding S_s/S_1 gets pointed at the hazard tool and the Chapter 22 maps instead of assuming the library is missing a feature. Also an explicit "do not add Fa/Fv here" with the reason.
2. **`from_site_adjusted` docstring** — says where to read S_MS/S_M1 from, rather than just what they aren't.
3. **`report()`** — if you pass `Ss=`/`S1=` (they're already accepted as optional metadata, and they enable the §11.6 `S1 >= 0.75` SDC override), they now appear in the report labelled *"reference only; not scaled"* and cited to §11.4.3, instead of being silently swallowed. Previously they only reached the spectrum metadata, so a calc package couldn't show them.

Plus two tests in `TestASCE722`: one pinning that `ASCE7_22` exposes no `Fa`/`Fv`/`Fa_override`/`Fv_override` (with the C11.4.3 quote in the docstring, so a future "helpful" re-add fails loudly with the reason), and one pinning that supplying `Ss`/`S1` changes S_DS/S_D1 by nothing.

Quality gate is clean: 507 tests pass, `ruff` clean, `mypy` clean on 27 files, and the 3.10 AST parse gate passes.

## One genuine limitation, stated plainly

The library has no USGS retrieval — S_s/S_1/S_MS/S_M1/MPRS are all user-supplied, for 7-22 as for the older editions. If the friction you're feeling is really "I have to go to a website and copy 22 numbers", a thin `from_usgs(lat, lon, site_class)` fetcher against the USGS Design Web Service would be a reasonable thing to build, and it would fetch real values rather than reconstructing removed ones. Say the word and I'll scope it — though note the skill's rule about network access: it would have to be an explicit opt-in call, never an implicit fetch.
