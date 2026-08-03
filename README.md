# codeSpectra

Codified seismic design response spectra as first-class Python objects.

Build a spectrum from site parameters, then scale it, reduce it by `R`,
envelope it against another code, floor it against a site-specific study,
export it to ETABS/SAP2000/OpenSees, or plot it — all through one interface.
Every derived quantity carries the clause that authorises it.

| Standard | Editions | Scope |
| --- | --- | --- |
| ASCE/SEI 7 | 7-10, 7-16, 7-22 | Design + MCEr spectra, vertical spectrum (7-16), SDC, ELF §12.8 |
| NEC-SE-DS (Ecuador) | 2015 | Elastic, inelastic and displacement spectra, DBF base shear §6.3 |
| NCh433 (Chile) | Of1996 Mod.2009 / DS 61-2011 | Elastic spectrum, `R*` reduction, `C` limits |

## Install

```bash
pip install -e ".[dev]"
```

Core runtime needs only numpy. `matplotlib` (`[plot]`) and `pandas`
(`[tabular]`) are optional extras.

## Quick start

```python
from codeSpectra import ASCE7_16

site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)

site.Fa, site.Fv          # 1.0, 1.7  — Tables 11.4-1 / 11.4-2, interpolated
site.SDS, site.SD1        # 1.0, 0.68
site.control_periods      # ControlPeriods(T0=0.136, Ts=0.68, TL=8)
site.seismic_design_category   # SeismicDesignCategory.D

design = site.design_spectrum()
design.at(1.0)            # 0.68 — exact, no interpolation error
```

Every parameter comes with its citation:

```python
print(site.report().to_text())
```

```
ASCE/SEI 7-16 seismic ground motion - Site Class D
--------------------------------------------------
  Ss = 1.5 g  (Mapped MCEr short-period acceleration)  [ASCE/SEI 7 7-16 §11.4.2 ...]
  Fa = 1 (Short-period site coefficient)  [ASCE/SEI 7 7-16 §11.4.4, Table 11.4-1 ...]
  SDS = 1 g  ((2/3) SMS)  [ASCE/SEI 7 7-16 §11.4.5, Eq. 11.4-3 ...]
  ...
  NOTE: Site Class D with S1 >= 0.2: ground motion hazard analysis required, unless ...
```

## NEC-SE-DS 2015

```python
from codeSpectra import NECSEDS2015
from codeSpectra.codes.nec import elf

site = NECSEDS2015(zone="V", soil="D", region="sierra")
site.Fa, site.Fd, site.Fs      # 1.20, 1.19, 1.28  — Tablas 3, 4, 5
site.control_periods           # T0=0.1269, Tc=0.6981, TL=2.856

elastic = site.elastic_spectrum()
reduced = site.inelastic_spectrum(R=8.0, phi_p=0.9, phi_e=1.0)

result = elf.base_shear(elastic, W=12_500.0, R=8.0, phi_p=0.9,
                        hn=24.0, structure_type="hormigon_porticos")
result.V, result.Ta, result.k
```

> **The ascending branch is off by default.** NEC-SE-DS restricts the
> `T <= T0` ramp to dynamic analysis, and within that to modes *other than the
> fundamental*. The design spectrum's plateau runs from `T = 0`. Pass
> `elastic_spectrum(include_ascending_branch=True)` only when evaluating
> higher modes.

## ASCE 7-22 takes different inputs

7-22 §11.4.3 reads `Ss`, `S1`, `SMS` and `SM1` **directly from the USGS
Seismic Design Geodatabase for the applicable site class**. The `Fa`/`Fv`
tables of 7-16 were removed, so there is no path from mapped `Ss`/`S1` plus a
site class to `SMS`/`SM1`. The two constructors mirror the two paths the
standard defines:

```python
from codeSpectra import ASCE7_22

# §11.4.5.1 multi-period — the default basis
site = ASCE7_22.from_mprs(periods, sa_mcer, site_class="CD", TL=8.0)

# §11.4.5.2 two-period — permitted only under §11.4.5 Exception 2
site = ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD")
```

## Spectra compose

`Spectrum` is a value object. Operations return new spectra and never mutate.

```python
design.scaled(1.5)                       # MCEr
design.reduced(R=8.0, Ie=1.0)            # inelastic
asce_spectrum.envelope(nec_spectrum)     # point-wise maximum
site_specific.floored_by(code, 0.80)     # the ASCE §21.3 80% floor
design.displacement(2.0)                 # Sd = Sa g (T/2pi)^2
```

Two concrete kinds share that interface: `AnalyticSpectrum` carries a
closed-form evaluator, so `at(T)` is exact at any period including right on a
branch corner; `TabulatedSpectrum` interpolates linearly in `T` and applies
the §11.4.5.1 beyond-10 s decay rule.

## Export

```python
from codeSpectra.export import to_etabs, to_opensees, to_csv, to_json

to_etabs(design, "spectrum.txt")                    # ETABS / SAP2000 function file
to_opensees(design, "spectrum.tcl", series_tag=1)   # timeSeries Path
to_json(design, "spectrum.json")                    # ordinates + full provenance
```

Every writer samples on a grid that lands exactly on the control periods, so
the exported curve reproduces the corners of the code figure instead of
chamfering them.

## Design principles

**Site-specific means site-specific.** Where a table defers to a study —
ASCE 7-16 Site Class E at `Ss >= 1.0`, any Site Class F, NEC soil type F — the
lookup raises `SiteSpecificRequired` carrying the triggering clause, rather
than inventing a coefficient. Interpolating *toward* an undefined cell raises
too. Two escape hatches exist, both explicit: `allow_site_specific_exception`
applies documented code exceptions, and `Fa_override`/`Fv_override` let an
engineer inject values from a completed study.

**Everything in g.** Every standard defines `Sa` as a fraction of gravity, so
that is the internal unit. Conversion happens only at the export boundary.

**Citations travel with values.** `ClauseRef` on every coefficient, control
period and spectrum; `report()` renders to text or Markdown for a calc package.

## Verification

Table values are transcribed from the standards themselves, not from memory:

- ASCE/SEI 7-16 Tables 11.4-1, 11.4-2, 11.9-1, 12.8-1, 12.8-2
- ASCE/SEI 7-10 Tables 11.4-1, 11.4-2
- ASCE/SEI 7-22 §11.4.2 site classes, §11.4.5.1 period list
- NEC-SE-DS 2015 Tablas 1, 3, 4, 5, 6, and §6.3.3 / §6.3.5
- NCh433 Tablas 4.2, 6.1, 6.2, 6.3, 6.4

The suite asserts every transcribed cell, checks the spectrum is continuous at
every control period across all zone x soil x region combinations, and covers
the branch equations, the site-specific triggers, and the ELF limits.

```bash
python -m pytest
```

## Not implemented

- Automatic retrieval of `Ss`/`S1` or MPRS ordinates from USGS — supply them.
- NCh433 Eq. 6-11, the alternative `R*` for wall-type buildings.
- ASCE 7 Chapter 21 site response analysis itself (the §21.3 floor is
  expressible via `floored_by`, but the study is yours to perform).
- Ecuadorian seismic hazard curves for essential structures (NEC §3.1.2).

## Licence

See [LICENSE](LICENSE).
