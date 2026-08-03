# codeSpectra

Codified seismic design response spectra as first-class Python objects.

Build a spectrum from site parameters, then scale it, reduce it by `R`,
envelope it against another code, floor it against a site-specific study,
export it to ETABS/SAP2000/OpenSees, or plot it — all through one interface.
Every derived quantity carries the clause that authorises it.

| Standard | Editions | Scope |
| --- | --- | --- |
| ASCE/SEI 7 | 7-10, 7-16, 7-22 | Design + MCEr spectra, vertical spectrum (7-16), SDC, ELF §12.8 |
| NEC-SE-DS (Ecuador) | 2015 | Elastic, inelastic and displacement spectra, DBF base shear §6.3, Tabla 19 poblaciones |
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

### Where `Z` comes from

**Tabla 19 ships with the library** — all 515 poblaciones, transcribed from the
standard. Look up by name, or by coordinates:

```python
from codeSpectra.codes.nec import Tabla19, Gazetteer, nec_site_from_poblacion

t = Tabla19.load()
t.by_name("Quito").Z          # 0.40  — case- and accent-insensitive
t.by_name("cuenca").Z         # 0.25

site = nec_site_from_poblacion(t.by_name("Guayaquil"), soil="D")
site.Z_g, site.eta            # 0.40, 1.80  — region derived from the province
```

`region` (and so `η`) is derived from the provincia via §3.3.1's own grouping,
including its explicit carve-out that Esmeraldas and Galápagos take the Sierra
value despite Esmeraldas being coastal. Two entries the standard does *not*
place — Santo Domingo de los Tsáchilas and "Zona no delimitada" — raise rather
than being guessed, and need `region=` supplied.

Thirty names are duplicated across provinces; `by_name` raises
`AmbiguousPoblacion` listing the candidates (and reporting whether they even
differ in `Z`) rather than silently picking one. Pass `provincia=` or `canton=`
to resolve.

#### Map-based lookup

This implements NEC §3.1.1's own fallback for a site that is not itself listed:

> *"Si se ha de diseñar una estructura en una población o zona que no consta en
> la lista … debe escogerse el valor de la población más cercana."*

```python
gaz   = Gazetteer.geonames()          # bundled coordinates, no download
match = t.nearest(lat=-0.30, lon=-78.50, gaz, max_distance_km=100)

match.poblacion.poblacion, match.distance_km, match.Z   # 'CONOCOTO', 2.8, 0.40
site = nec_site_from_poblacion(match, soil="D")
```

NEC publishes names, not positions, so the coordinates come from **GeoNames**,
matched to Tabla 19 once and shipped as a derived table — 446 of the 506
resolvable `(población, provincia)` pairs, 88%. Every spot-checked city lands
within 2 km of its true position.

> Place coordinates derived from the [GeoNames geographical
> database](https://www.geonames.org/), used under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reproduce this
> attribution wherever you publish results built on them —
> `GEONAMES_ATTRIBUTION` holds the text.

You can supply your own instead — a plain `{name: (lat, lon)}` dict, or
`Gazetteer.from_file()` for any GeoJSON point layer.

Three things worth knowing:

- **Entries are province-qualified.** 24 names in Tabla 19 occur in more than
  one province and 20 of those carry a *different* `Z`, so a gazetteer keyed
  on name alone would resolve some sites to the wrong design value.
  `Gazetteer.get(name, provincia)` never leaks across provinces.
- **One pair is deliberately absent.** `PUEBLO NUEVO` in Guayas exists in two
  cantones with `Z` of 0.40 and 0.50; a single coordinate could not resolve
  it safely, so it is omitted rather than guessed.
- **Coverage is not complete.** `t.covered_by(gaz)` reports what a gazetteer
  can place, and `max_distance_km` refuses a match too far away to be *la
  población más cercana* in any useful sense. The report reminds you to check
  Figura 1 for a zone boundary between the site and the matched town.

You can also just supply `Z` yourself, straight from Figura 1.

#### External PSHA models

Optionally, `Z` can be read off a published PSHA. codeSpectra ships **no
third-party hazard data**; it ships the reader, and you point it at layers you
hold or let it fetch them from the publisher on an explicit call:

```python
from codeSpectra.codes.nec import ContourHazardMap, nec_site_from_hazard

hazard = ContourHazardMap.from_palacios_2023(path="CurvasNivelhmapmeanPGA475TR_2.js")
# or, an explicit network call — never implicit:
# hazard = ContourHazardMap.from_palacios_2023(download=True)

est = hazard.pga_at(lat=-0.1807, lon=-78.4678)   # Quito
est.pga, est.band, est.distance_km               # 0.480, (0.4, 0.5), 6.5 km

site = nec_site_from_hazard(est, soil="D", region="sierra")
```

The [Palacios, Celi & Poveda (2023)](https://github.com/ppalacios92/SeismicHazard2023_poe0.1_50y)
map is mean PGA at 10% PoE in 50 years — the same 475-year hazard level as
NEC's design earthquake, which is why it maps onto `Z` at all.

Three things the API enforces, because this is not code data:

- **It is a band, not a number.** The published layers are iso-lines at 0.1 g
  steps; `est.pga` interpolates between them but is never more precise than
  that interval. `est.band` and `est.distance_km` are the honest outputs.
- **Distant points are refused.** A query far outside the contours still
  yields a plausible-looking figure from nearest-line interpolation — the
  Galápagos return 0.449 g from geometry 1,070 km away. Those are flagged
  `reliable=False`, and `nec_site_from_hazard` raises rather than building a
  design site from one (`allow_unreliable=True` to override).
- **Provenance leads every report.** Any site built this way carries a note,
  first in the list, saying `Z` did *not* come from the NEC zone map, plus the
  authors' citation and the licence position.

`region` still has to be given: NEC's `η` follows provincial boundaries
(Costa / Sierra / Oriente, with Esmeraldas and Galápagos taking the Sierra
value), which a PGA contour map cannot resolve.

> **Licence.** That repository publishes no `LICENSE` file, so it is
> all-rights-reserved by default and citation alone does not grant
> redistribution. That is exactly why nothing is vendored here. If the authors
> add a licence (CC-BY-4.0 would make attribution sufficient), bundling the
> layers for offline use becomes a one-line change.

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
- NEC-SE-DS 2015 Tablas 1, 3, 4, 5, 6, 19, and §6.3.3 / §6.3.5
- NCh433 Tablas 4.2, 6.1, 6.2, 6.3, 6.4

The suite asserts every transcribed cell, checks the spectrum is continuous at
every control period across all zone x soil x region combinations, and covers
the branch equations, the site-specific triggers, and the ELF limits.

```bash
python -m pytest
```

## Not implemented

- Automatic retrieval of `Ss`/`S1` or MPRS ordinates from USGS — supply them.
- Any bundled hazard data. The Ecuador PSHA reader ships without layers; see
  [Where `Z` comes from](#where-z-comes-from).
- Full coordinate coverage of Tabla 19 — the bundled GeoNames match reaches
  88% of resolvable pairs.
- NCh433 Eq. 6-11, the alternative `R*` for wall-type buildings.
- ASCE 7 Chapter 21 site response analysis itself (the §21.3 floor is
  expressible via `floored_by`, but the study is yours to perform).
- Ecuadorian seismic hazard curves for essential structures (NEC §3.1.2).

## Licence

See [LICENSE](LICENSE).
