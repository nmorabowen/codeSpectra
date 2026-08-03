Done — `region=` is now optional on `nec_site_from_hazard()` and defaults to the region of the province the estimate's own coordinates fall in.

```python
site = nec_site_from_hazard(est, soil="D")                 # eta from the coordinates
site = nec_site_from_hazard(est, soil="D", region="costa") # explicit still wins
```

And the lookup is usable on its own:

```python
from codeSpectra.codes.nec import provincia_at, region_at

provincia_at(lat=-0.1807, lon=-78.4678).provincia   # 'Pichincha'
region_at(lat=-1.4924, lon=-77.9962)                # 'oriente'  (Puyo)
```

## What ships

`src/codeSpectra/codes/nec/tables/provincias_osm.json` (347 kB) — the 24 Ecuadorian provinces as OpenStreetMap `admin_level=4` relations, fetched once from Overpass and bundled, so it works offline exactly like the GeoNames gazetteer. The new module `provincias.py` does point-in-polygon against it, then hands the province to the existing `region_for_provincia()` so the §3.3.1 grouping stays the single source of truth for costa/sierra/oriente.

One detail worth knowing about how I built it: simplification is applied **per OSM way**, not per assembled ring. Adjacent provinces share the same way objects in OSM, so thinning at the way level keeps their common boundary bit-identical — no slivers or overlaps open up along a border, which is what you get if you simplify each province polygon independently.

## Licence — please read this before publishing anything built on it

OSM is **ODbL v1.0**, not MIT. The bundled table is a *Derivative Database* under that licence, so attribution **and share-alike** travel with it, and it is not covered by codeSpectra's own MIT licence. I have:

- put the notice in the JSON header and in a new `OSM_ATTRIBUTION` constant, mirroring how `GEONAMES_ATTRIBUTION` already works;
- made every derived region add that notice to the site's report automatically (a spectrum is a Produced Work — it needs the attribution but not the share-alike);
- added a licence table to the README stating plainly that two bundled data files are not MIT and what each one requires.

If you'd rather not carry a share-alike obligation in the repo at all, the alternative is the same shape as the Palacios layers: ship the reader, not the data. Say the word and I'll flip it to a `ProvinceBoundaries.from_file()` / `download=True` model.

## Three limitations I did not paper over

**The boundaries are today's; §3.3.1's groupings are not.** This bit immediately. Cantón **La Concordia** is listed under *Esmeraldas* in Tabla 19, but on the current map it belongs to *Santo Domingo de los Tsáchilas* — a real territorial change since 2015. It showed up as exactly three towns (`LA VILLEGAS`, `MONTERREY`, `PLAN PILOTO`) where the polygons disagree with the gazetteer. Since §3.3.1 places Santo Domingo in **no** region at all, a site there raises `InvalidInput` and asks you to pass `region=` after deciding which province the standard meant — it does not guess. There's a test pinned to that disagreement so the caveat can't go stale silently.

**The geometry is simplified** to ~110 m, to keep the table at 347 kB rather than 2.6 MB. I measured the cost against the full-resolution OSM geometry over 20,000 random points inside Ecuador: **0.10%** change province, **0.035%** change region — all of them within about 100 m of a boundary. `match.boundary_distance_km` tells you how close you are and `match.near_boundary` flags it, and the report warns when it matters.

**Outside Ecuador it refuses**, naming the nearest province and the distance, rather than returning the closest thing it found. Swapped lat/lon raises rather than answering. Points that miss by under 2 km — a coastal fill, a simplified river border — are attributed to the nearest province with `inside=False` recorded; `tolerance_km=0` demands strict containment.

This resolves `η` only. `Z` still comes from Figura 1 or Tabla 19 — provincial geometry is not the zone map.

## Verification

- 21 provincial capitals plus Esmeraldas, Galápagos and Santo Domingo — all 24 provinces — land in the right province and region.
- Cross-checked against the GeoNames gazetteer, which was derived completely independently: **442 of 446** points agree. Of the four, three are the La Concordia towns above and one (`GENERAL FARFAN`) sits 0.55 km across a simplified river border, which the 2 km tolerance absorbs.
- Full gate green: **582 tests pass** (67 new), `ruff check` clean, `mypy --strict` clean.
