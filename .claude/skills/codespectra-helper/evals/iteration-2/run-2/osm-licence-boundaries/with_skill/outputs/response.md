Done — `region=` is now optional and derived from the coordinate. One thing about the OSM licence you should decide on before this ships, at the bottom.

```python
site = nec_site_from_hazard(est, soil="D")     # region derived
region_from_coordinates(lat=-2.1900, lon=-79.8875)   # 'costa'
ProvinceBoundaries.load().locate(0.9592, -79.6539)   # Esmeraldas -> 'sierra'
```

## I checked whether boundaries were actually needed first

You already had a cheaper route to the province: `Tabla19.nearest()` plus the
bundled GeoNames gazetteer gives you a town, and every Tabla 19 row names its
provincia. No new data, no new licence. So I measured it before building
anything — nearest-town region vs point-in-polygon region on a 0.02° grid over
Ecuador:

- **93.7% agreement.** Nearest-town gets the region wrong on **6.3% of the land
  area**, with the disagreeing points a median 25 km from the nearest listed
  town.
- The errors are exactly where towns are sparse: the Sierra/Oriente flank
  (1,577 cells) and the Sierra/Costa flank (1,017 cells).
- The worst class is a Sierra site read as Costa — `eta` 2.48 → 1.80, a **27%
  under-estimate of the whole plateau**.

That is not an acceptable silent failure mode, so the polygons earn their
place. Worth knowing that the cheap version existed and was measured, though.

## What shipped

- **`codes/nec/provincias.py`** — `ProvinceBoundaries` (point-in-polygon,
  ~1 ms/query) and `region_from_coordinates()`. The province→region mapping
  stays in the existing `region_for_provincia()`, since that part is §3.3.1's
  own text; only the geometry is new.
- **`tables/provincias_osm.json`** — 24 provinces from OSM `admin_level=4`,
  446 KiB, offline, reaches the wheel (checked).
- **`scripts/extract_province_boundaries.py`** — rebuilds it from Overpass and
  re-runs every validation. Reproduces the shipped file byte-identically.
- `nec_site_from_hazard(region=None)` derives, and records how in the
  provenance note.

Extraction traps that mattered:

- The Overpass area query also returns **Nariño (CO-NAR)**, which shares a
  border way with Carchi. Filtered on `ISO3166-2 ^EC-`, and there is a test
  asserting it stayed out.
- 158k raw vertices → 22k at a 100 m Douglas-Peucker tolerance. I checked the
  cost: simplification changes the province answer on 0.086% of grid cells,
  all on boundaries, and changes the *region* answer on none of them.
- Cross-checked against a completely independent dataset: **449 of 453** Tabla
  19 towns the GeoNames gazetteer can place fall inside the province Tabla 19
  itself names. Both the exceptions are real and pinned in tests (below).

## Three things that now raise instead of guessing

- **Santo Domingo de los Tsáchilas.** §3.3.1 does not assign it a region, so
  `region_from_coordinates` raises there. Pass `region=` explicitly.
- **Outside Ecuador.** Previously `nec_site_from_hazard` happily returned a
  `"sierra"` site for a coordinate in Peru. Now `ProvinceNotFound`.
- I also found that `region_for_provincia("Santo Domingo de los Tsáchilas")`
  gave "Unknown Ecuadorian province" rather than the explanatory message —
  Tabla 19 abbreviates it `STO.` and every other source spells it out. Both
  spellings now hit the real message.

**La Concordia.** Monterrey, La Villegas and Plan Piloto are listed under
Esmeraldas in Tabla 19, but the canton was transferred to Santo Domingo in
2013 and OSM has them there. Neither source is wrong; they are of different
dates. This one bites: §3.3.1 gives Esmeraldas the *Sierra* eta and does not
assign Santo Domingo at all. Pinned in a test so it can't drift silently.

Near a provincial line (<5 km) the note names the neighbour and what `eta`
would be on the other side — a Los Ríos site 0.4 km from Bolívar is 1.80 vs
2.48. Distance is measured to the nearest *other* province, not to the
containing polygon's own outline, or every coastal site would be flagged
against the ocean.

## The licence — this is the part I'd want you to sign off

OSM is **ODbL 1.0**, not a permissive licence. It does allow redistribution,
which is why I bundled it rather than shipping a reader like the Palacios PSHA
map — but it is **share-alike**, and codeSpectra is MIT. That is a real
change to the package's licensing position:

- `provincias_osm.json` stays ODbL inside an otherwise-MIT wheel;
- anyone redistributing codeSpectra redistributes an ODbL database;
- any *produced work* — a report or spectrum whose region came from this file —
  must carry the OpenStreetMap attribution.

All three are wired up: the JSON carries its own licence header, `LICENSE` now
has a third-party section spelling out the obligation, and `OSM_ATTRIBUTION`
is emitted automatically in the provenance note of every affected report. It
also says plainly that deleting the file and passing `region=` explicitly
removes the obligation entirely — nothing else depends on it.

If you would rather not take on share-alike at all, **geoBoundaries ADM1 for
Ecuador is CC0**, and swapping is a matter of re-running the extractor against
a different source. I did not use it because its ISO codes are wrong (Cotopaxi
and Chimborazo are both tagged `EC-H`), its provenance is a bare Wikimedia
Commons file URL with no licence detail, and it is a 2011 vintage — so it would
miss the La Concordia transfer that OSM has. Better data, encumbered licence,
versus cleaner licence, worse data. I picked the data and documented the cost,
but it is your call and it is one file to change.

## Gate

`pytest` 582 passed · `ruff` clean · `mypy --strict` clean · 3.10 AST parse
clean · JSON confirmed present in the built wheel. 69 new tests. I also
updated the skill's §9 licensing table, since "ship the reader unless the
licence permits redistribution" now has a case where the licence permits it
but still costs something.
