# Notes — deriving the NEC region from coordinates

## The request, and where I diverged

Asked: make `region=` optional on `nec_site_from_hazard()` by resolving lat/lon
to an Ecuadorian province, and **bundle OpenStreetMap province boundaries** so
it works offline like the GeoNames gazetteer.

Delivered the feature; declined to bundle OSM. Two reasons, in order of weight.

### 1. ODbL share-alike is a licence decision for the project owner

Verified against <https://osmfoundation.org/wiki/Licence>: OSM data extracted
after Sept 2012 is Open Database License 1.0 — **share-alike**. A province
boundary extract is a Derivative Database. Bundling one into codeSpectra would:

- place an ODbL database inside every install and redistribution of an
  MIT-licensed wheel (`pyproject.toml` → `license = { file = "LICENSE" }`, MIT);
- oblige any downstream derived database to be offered under ODbL;
- require `(c) OpenStreetMap contributors` attribution on derived output.

This is categorically unlike the already-bundled GeoNames gazetteer, which is
CC BY 4.0 — attribution-only, no reach-through. The repo is visibly careful
here already: `hazard.py`'s module docstring opens with "No third-party data
ships with codeSpectra", `PALACIOS_2023.licence` records an all-rights-reserved
position rather than vendoring, and `test_no_data_is_bundled` asserts the
package ships no `.geojson`. Silently adding a share-alike database would cut
against a stance the codebase states explicitly in three places.

Flagged it, offered two bundle-able alternatives verified for licence terms:
- Natural Earth admin-1 — public domain, attribution not required
  (<https://www.naturalearthdata.com/about/terms-of-use/>);
- geoBoundaries ADM1 — CC BY 4.0, same footing as the gazetteer.

Vendoring either is a small follow-up now the loader exists.

### 2. The polygons turned out to be unnecessary

Tabla 19 names the **provincia** of every one of its 515 poblaciones, and
`gazetteer_geonames.json` already places 453 of them. So point → provincia
needs no new data: the nearest listed poblacion yields a province taken from
the code's own table, and `region_for_provincia()` maps it to a region per
§3.3.1. This is the §3.1.1 nearest-town rule the package already applies to
`Z`, redirected at the provincia column.

Provenance is arguably stronger than a third-party polygon would give: the
province assignment is code data, not an external map.

## What was built

**`src/codeSpectra/codes/nec/regions.py` (new)**

- `region_at(lat, lon, *, gazetteer=None, boundaries=None, max_distance_km,
  contest_ratio)` → `RegionMatch`.
- `RegionMatch` — region, provincia, the poblacion used, distance, the nearest
  poblacion of a *different* region, `eta`, `contested`, and a `report()`
  carrying ClauseRefs for §3.1.1 and §3.3.1.
- `RegionUndetermined(InvalidInput)` for the three refusal cases.
- `ProvinceBoundaries` — point-in-polygon over a user-supplied GeoJSON layer
  (Polygon/MultiPolygon, holes honoured, qgis2web `.js` wrapper stripped).
  Ships the reader, not the data — same posture as `ContourHazardMap`.
- `BOUNDARY_DATA_NOTE` — the ODbL position and the lighter alternatives,
  in-package rather than only in the README.

**`hazard.py`** — `region: str = "sierra"` → `region: str | None = None`; added
`boundaries=` passthrough; derivation appended to the provenance note.

**`nec/__init__.py`** — exports the new names, plus `GEONAMES_ATTRIBUTION`
which was defined but never re-exported.

**`README.md`** — two new subsections; the old "`region` still has to be given"
paragraph removed; "Not implemented" notes no bundled polygons.

## Design calls worth recording

**`contested` had to be a ratio, not a margin.** First cut flagged a site when
another region's nearest town was within 25 km. Measured against all 453
covered poblaciones: **30 false positives** — towns queried at their *own*
coordinates (distance 0.0, answer certain) flagged merely because a
cross-region town sat 8–20 km away. Several Bolívar, Cañar, Cotopaxi and El Oro
towns do. The real question is whether the site is near-*equidistant*, since
the implied boundary runs midway. Switched to
`distance > 0.75 * rival_distance`. Result: 0 false positives at town
coordinates, genuine mid-boundary points (e.g. −4.5, −79.1) still flag.

**Refuse rather than guess, in three cases.**
- Contested → `nec_site_from_hazard` raises. η is 1.80 (costa) vs 2.48
  (sierra), a 38% plateau swing; too large to resolve by coin-flip. `region=`
  is the escape hatch, mirroring the existing `allow_unreliable` pattern.
- Beyond `max_distance_km` → raises. **Tabla 19 lists no Galápagos poblacion**;
  the nearest mainland town is 1,046 km away. The message names §3.3.1's own
  answer (`region="sierra"`) instead of extrapolating.
- Province §3.3.1 never placed (Santo Domingo de los Tsáchilas, "Zona no
  delimitada") → raises. Deliberately keyed on the *overall* nearest town, not
  the nearest *placeable* one: falling through would silently relocate a Santo
  Domingo site into a neighbouring province. Tested explicitly.

**`max_distance_km = 120`.** Sampled a grid over continental Ecuador: worst
in-country distance to a covered poblacion is ~70 km (Taisha, Morona
Santiago). 120 km accepts the whole mainland, rejects Galápagos (1,046 km) and
Bogotá (584 km).

**Avoided a stale-set bug.** First version hardcoded the unplaced province
names in `regions.py`. If someone added an entry to
`poblaciones.AMBIGUOUS_PROVINCES`, `region_for_provincia` would raise inside
the scan loop and break *every* query. Replaced with try/except around the
call, so the two cannot drift.

## Behaviour change to flag

`test_unreliable_can_be_forced` called `nec_site_from_hazard` on a Galápagos
point with no `region=`, previously silently defaulting to `"sierra"`. It now
raises, correctly, because the region cannot be derived there. Updated the test
to pass `region="sierra"` with a comment naming §3.3.1. Renamed
`test_region_is_not_guessed_from_coordinates` →
`test_explicit_region_overrides_any_derivation`: its assertion still holds and
is still worth keeping, but its old name asserted the opposite of the feature.

## Quality gate

| Check | Result |
|---|---|
| `pytest` | **567 passed**, up from 512 baseline (55 new); 0 failures |
| `ruff check src tests` | **All checks passed** |
| `mypy src` (strict) | **Success: no issues found in 28 source files** |
| `mypy tests` | 4 new lines, all the pre-existing `import-untyped` class affecting every test file (no `py.typed` marker); diffed against baseline to confirm no new real error |
| cp1252-safe reports | asserted for `region_at` and the derived-region site |
| `region_at` cost | ~2.5 ms/call |

Correctness check embedded as a test rather than a one-off: all 453 covered
poblaciones, queried at their own coordinates, return their own province —
0 mismatches.

Nothing committed; nothing downloaded into the repo; no network access at
runtime.
