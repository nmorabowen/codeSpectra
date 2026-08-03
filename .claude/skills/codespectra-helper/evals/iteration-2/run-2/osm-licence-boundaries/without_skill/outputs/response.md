Done — `region=` is now optional on `nec_site_from_hazard()`. But I did **not** bundle OpenStreetMap boundaries, and I want to be upfront about why before the rest.

## The OSM part is a licence decision, not a technical one

OSM data is [ODbL 1.0](https://osmfoundation.org/wiki/Licence), which is **share-alike**. A province-boundary extract is a Derivative Database, so bundling one into codeSpectra would:

- put an ODbL database inside every install and every redistribution of an MIT-licensed wheel,
- oblige anyone who builds a derived database from it to publish theirs under ODbL,
- require `(c) OpenStreetMap contributors` on anything produced from it.

That is materially different from the GeoNames gazetteer you already ship: CC BY 4.0 is attribution-only, with no reach-through. Quietly stapling a share-alike database into your package is exactly the kind of thing that should be your call, not a side effect of a feature request — so I stopped short of it.

Two sources would give you the same polygons on lighter terms, if bundling is what you want:

- **[Natural Earth admin-1](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/)** — public domain, attribution not even required. Cleanest fit.
- **[geoBoundaries ADM1](https://www.geoboundaries.org/)** — CC BY 4.0, i.e. exactly the footing the gazetteer is already on.

Say the word and I'll vendor either one; it's a small change now that the loader exists.

## What I built instead — and it needs no new data at all

It turns out you already ship everything required. Tabla 19 names the **provincia** of all 515 poblaciones, and the GeoNames gazetteer already places 453 of them. So the nearest listed poblacion hands back a province *read off the code's own table*, which `region_for_provincia()` then maps to a region per §3.3.1. That is the same §3.1.1 nearest-town rule you already use for `Z`, pointed at the provincia column instead of the `Z` column — so the provenance story is arguably better than a third-party polygon would give you.

```python
from codeSpectra.codes.nec import region_at

m = region_at(lat=-0.1807, lon=-78.4678)     # Quito
m.provincia, m.region, m.eta                 # 'PICHINCHA', 'sierra', 2.48
m.poblacion.poblacion, m.distance_km         # 'NAYON', 4.0
```

and the thing you actually asked for:

```python
site = nec_site_from_hazard(est, soil="D")   # no region= needed
```

The derivation is recorded in the site's provenance note (which poblacion, how far, which clause). Passing `region=` explicitly still wins outright and suppresses the derivation entirely.

**Verification that matters:** queried at its own coordinates, every one of the 453 covered poblaciones comes back with the province Tabla 19 assigns it — 0 mismatches. That's a test, not a spot-check. Quito/Guayaquil/Cuenca/Manta/Machala/Ambato/Coca/Tena/Esmeraldas all resolve correctly, including the §3.3.1 trap where Esmeraldas is coastal but takes the *sierra* η of 2.48.

## Three cases refuse rather than guess

Proximity is not containment, so I did not let it pretend otherwise:

- **Contested sites.** `RegionMatch.contested` flags a site close to equidistant between two regions' nearest towns. `nec_site_from_hazard` **raises** there — η is 1.80 vs 2.48 across the costa/sierra line, a 38% swing in the plateau, far too large to guess. Pass `region=` to resolve.
- **Outside the listed territory.** Tabla 19 lists **no Galápagos poblacion**, and the mainland is 1,000 km away. This raises `RegionUndetermined` naming §3.3.1's own answer (`region="sierra"`) rather than extrapolating. Heads up: this is the one behaviour change — your existing Galápagos test now needs `region="sierra"` passed, which I've done.
- **Provinces §3.3.1 never placed.** Santo Domingo de los Tsáchilas and "Zona no delimitada" raise, and are *not* silently reassigned to a neighbouring province. I tested that specific failure mode.

One design note: I first wrote `contested` as an absolute distance margin and it was wrong — it flagged 30 towns sitting *exactly on their own coordinates* just because another region's town was 10 km away. The right question is whether the site is near-equidistant, so it's now a ratio test (`DEFAULT_CONTEST_RATIO = 0.75`). False positives went to zero while genuine mid-boundary points still flag.

## The polygon route is there when you want it

`ProvinceBoundaries` does a real point-in-polygon test (rings + holes, Polygon/MultiPolygon) against any layer you hold — Natural Earth, geoBoundaries, or an OSM extract you keep in *your own* project, where ODbL is yours to accept:

```python
b = ProvinceBoundaries.from_file("provincias.geojson", name_property="name")
region_at(lat=-0.1807, lon=-78.4678, boundaries=b).region     # 'sierra'
nec_site_from_hazard(est, boundaries=b)                       # passed through
```

Ships the reader, not the data — same posture as `ContourHazardMap`. `BOUNDARY_DATA_NOTE` carries the licence summary in-package, and a test asserts no polygon layer is ever bundled.

## Gate

567 tests pass, up from 512 — 55 new, none failing. `ruff check` clean, `mypy --strict src` clean. Reports stay cp1252-safe. `region_at` costs ~2.5 ms.

Files: `src/codeSpectra/codes/nec/regions.py` (new), `hazard.py`, `nec/__init__.py`, `tests/test_nec_regions.py` (new), `tests/test_nec_hazard.py`, `README.md`.
