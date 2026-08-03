# Notes — deriving the NEC region from coordinates via OSM province boundaries

## What I did

**New module `src/codeSpectra/codes/nec/provincias.py`** — a boundary reader
and point-in-polygon lookup:

- `ProvinceBoundaries` (`Sequence[Province]`) with `.load()` (bundled, cached),
  `.from_geojson()`, `.from_file()`, `.provincia_at()`, `.region_at()`,
  `.by_name()`.
- `ProvinceMatch` frozen dataclass carrying `provincia`, `iso_3166_2`,
  `margin_km`, `nearest_other`, `nearest_other_km`, `source`, a `.region`
  property, `.near_region_boundary(tolerance_km=5)` and `.report()`.
- `BoundarySource` provenance record (mirrors the existing `HazardSource`).
- `ProvinceNotFound`, `AmbiguousProvince` exceptions.
- Module-level `provincia_at()` / `region_at()` / `region_provenance_note()`
  over the bundled data.

**New data `src/codeSpectra/codes/nec/tables/provincias_osm.json`** — 24
Ecuadorian provinces, 228 kB, from OSM `admin_level=4`. Carries its own
`source` / `url` / `licence` / `attribution` / `extracted` / `note` header, the
same shape as `gazetteer_geonames.json`.

**New script `scripts/build_provincias_osm.py`** — the complete, re-runnable
derivation (Overpass → ring assembly → Douglas-Peucker → round → validate).
ODbL compliance needs the derivation to be reproducible, so this belongs in the
repo rather than in the skill's scratch scripts. Its raw download cache is
gitignored.

**New `src/codeSpectra/codes/nec/_geo.py`** — `haversine_km`,
`point_to_polyline_km`, `point_in_ring`. The first two were already duplicated
between `hazard.py` and `poblaciones.py` (with two copies of the Earth radius);
they now live in one place and both callers import from it. No behaviour
change.

**`nec_site_from_hazard(region=...)` now defaults to `None`** and derives the
region from `estimate.latitude/longitude`, appending a provenance note that
names the province, the margin, the §3.3.1 grouping and the OSM attribution.

**`region_for_provincia`** gained an alias table: Tabla 19 abbreviates
`STO DOMINGO DE LOS TSACHILAS`, every other source (OSM included) spells
`SANTO` out, so without the alias every Santo Domingo point would have failed
with "Unknown Ecuadorian province" instead of the correct
"§3.3.1 does not assign a region".

**Licensing paperwork**: `LICENSE` gained a "Third-party data" section (MIT
text untouched above it); `codes/nec/tables/LICENSE-OpenStreetMap.txt` states
the ODbL position; `pyproject.toml` package-data widened to `**/tables/*.txt`
so the notice reaches the wheel; README gained a section and a revised
"Licence" heading.

## Why these choices

**Bundling OSM at all.** The skill's §9 rule is "ship the reader, not the data,
unless the licence clearly permits redistribution". ODbL *does* clearly permit
it — that is the difference from the Palacios 2023 layers, which have no
licence file and so are all-rights-reserved. So bundling is legitimate. What it
is *not* is free: ODbL is share-alike, so the wheel stops being uniformly MIT.
I did the work and made the consequence explicit in three places rather than
either refusing the request or vendoring the data silently. The response leads
with it and offers the CC BY / public-domain alternatives, since the reader is
dataset-agnostic.

**The region is still the code's own rule.** §3.3.1 groups provinces, and
`region_for_provincia` already encoded that grouping. The boundaries only
answer "which province", which NEC does not publish — structurally identical to
the GeoNames gazetteer supplying coordinates for Tabla 19's names. So no
invented value enters: the report separates the two provenances.

**Refusals preserved.** Santo Domingo de los Tsáchilas raises (§3.3.1 never
placed it), and a point outside Ecuador raises rather than snapping to the
nearest province. The temptation with "stop making me pass region" is to make
*something* come back always; that would have quietly resolved the one case the
standard genuinely leaves open.

**Sliver handling.** Shared borders are simplified once per province and
Douglas-Peucker is not invariant to where a closed ring is cut, so the two
sides of a border can differ by ~200 m, leaving hairline gaps and overlaps. I
resolved this by the *decision* rather than the geometry: within 0.5 km of a
line, if every candidate province is in the same region the answer is
unaffected and the nearest is returned; if they differ, `AmbiguousProvince` is
raised naming both. Measured: walking the whole Guayas ring (427 samples), 3
were ambiguous and 424 resolved; on a 0.05° grid over the mainland, 8,084
resolved, 123 fell in the unplaced province, 0 ambiguous, 0 silent failures.

**The real defect underneath.** `nec_site_from_hazard` had `region="sierra"` as
a *silent default*, so an omitted argument gave a Guayaquil site η = 2.48
instead of 1.80 — 38% high, with nothing in the report saying so. The skill's
§14 note ("treat a confident request as a prompt to look for the real problem
underneath it") is what made me check the old signature rather than just adding
a lookup.

## What went wrong during the build

`overpass.osm.ch` carries **Switzerland only**. It returned HTTP 200 with a
well-formed JSON body containing zero elements for 16 of the 24 provinces. My
first fetcher validated only that the response parsed, so it cached 16 empty
files. Had I not counted elements, the shipped dataset would have had 16
provinces with no geometry — and the failure mode would have been "point not in
any province" for two-thirds of Ecuador. Fixed by restricting the mirror list
to world-coverage instances and treating an empty element list as an error, in
both the throwaway fetcher and `scripts/build_provincias_osm.py`. This is
exactly the §7 lesson (validate the extraction; assert on counts) in a
non-PDF setting.

The public Overpass instances also 429/504 heavily; the script retries across
mirrors and caches per relation, so a rebuild is incremental.

## Deliberately not done

- No new `nec_site_from_coordinates()` convenience constructor. `Z` still has to
  come from Figura 1 / Tabla 19 / a study, and the two existing entry points
  already cover the paths; adding a third would be scope creep.
- `nec_site_from_poblacion` still derives the region from the Tabla 19 row's
  *provincia*, not from coordinates. That path is exact code data — routing it
  through geometry would make it worse.
- No change to the ZONA NO DELIMITADA handling. Tabla 19 lists those rows;
  OSM's present-day boundaries assign the territory to the provinces that
  absorbed it after 2015. Documented as a divergence rather than papered over.

## Quality gate — all clean

```
python -m pytest -q                      583 passed
python -m ruff check .                   All checks passed
python -m mypy src/codeSpectra           Success: no issues found in 29 source files
python -c "ast.parse(..., (3,10))"       py3.10 parse OK  (src/ and scripts/)
python -m pytest --doctest-modules src   9 passed
python -m build --wheel                  wheel contains provincias_osm.json
                                         AND LICENSE-OpenStreetMap.txt
```

The wheel check mattered here: `package-data` was `**/tables/*.json` only, so
the ODbL notice would have been dropped from every installed copy — invisible
until after install, and precisely the compliance artefact that must not go
missing.

Build artefacts (`dist/`, `scripts/_osm_cache/`) were removed after
verification. Nothing was committed.
