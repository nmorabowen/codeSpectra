# Notes — derive NEC region from coordinates via bundled OSM province boundaries

## Setup

Ran the required Step 0: `git reset --hard 198bbad`, `rm -rf .claude/skills`.
Confirmed `src/codeSpectra/codes/` exists and `.claude/skills` is gone. All Python
run with `PYTHONPATH=src` from this worktree.

The `.claude/skills/**` deletions from Step 0 show in `git status` but are
deliberately **excluded** from `changes.patch` — they are the harness's setup
step, not part of this change. The patch is `git diff -- README.md src tests`
after `git add -N` on the three new files; the index was restored with
`git reset` afterwards. Nothing was committed.

## What I did

### 1. Fetched the data (Overpass)

First attempt — a bare global `["ISO3166-2"~"^EC-"]` regex on `admin_level=4` —
returned HTTP 504. Narrowing with an area filter first works:

```
[out:json][timeout:900];
area["ISO3166-1"="EC"]["admin_level"="2"]->.ec;
rel(area.ec)["boundary"="administrative"]["admin_level"="4"]["ISO3166-2"~"^EC-"];
out geom;
```

8.5 MB, 24 relations — exactly Ecuador's 24 provinces. Without the
`ISO3166-2` filter the area query also picks up Nariño (CO-NAR), the
Colombian department across the border, so that filter is load-bearing.

The query, retrieval date, simplification tolerance and coordinate precision
are all recorded in the JSON header, so the table can be regenerated. I did not
add a build script — the repo has no `scripts/` precedent (the GeoNames
gazetteer records its method in its header the same way) and the header is
sufficient to reproduce.

### 2. Simplified per OSM *way*, not per ring

This is the one non-obvious build decision. Adjacent provinces share the same
OSM way objects along their common border. If you assemble each province's
rings first and then simplify them independently, Ramer-Douglas-Peucker makes
different choices on each side and opens slivers and overlaps along every
internal border — points near a boundary then land in two provinces or none.
Simplifying each distinct way **once**, before assembling rings, keeps shared
borders bit-identical. Rounding to 4 dp is likewise applied per way, so it is
deterministic and does not reintroduce divergence.

Tolerance 0.001 deg (~110 m), coordinates to 4 dp (~11 m): 347 kB versus
2.6 MB unsimplified. That is ~10x the other bundled tables, which felt like the
right ceiling for a package data file.

### 3. Measured what the simplification costs

Rather than assert "accurate enough", I built the unsimplified table too and
compared classification over 20,000 uniformly random points that fall inside
Ecuador (seeded, reproducible):

- province differs: 20 / 20,000 = **0.100%**
- region differs:    7 / 20,000 = **0.035%**

All of them within roughly 100 m of a boundary. That figure is quoted in the
README and the module docstring, and it is why `ProvinceMatch` carries
`boundary_distance_km` and `near_boundary` rather than pretending the answer is
crisp.

### 4. Wired it up

- New `src/codeSpectra/codes/nec/provincias.py`: `ProvinceBoundaries`,
  `ProvinceMatch`, `ProvinceNotFound`, `provincia_at()`, `region_at()`,
  `OSM_ATTRIBUTION`. Point-in-polygon is crossing-number on numpy arrays with a
  bbox prefilter; distance-to-boundary uses the same local equirectangular
  projection `hazard.py` already uses. ~0.34 ms/query, 7 ms to load.
- `region_at()` delegates to the existing `region_for_provincia()` so §3.3.1's
  grouping is not duplicated. The new module owns *geography*; `poblaciones.py`
  keeps owning the *code rule*.
- `nec_site_from_hazard(region=None)` derives from `estimate.latitude/longitude`
  and appends the derivation (province, distance, near-boundary warning) plus
  the OSM notice to the provenance note. An explicit `region=` skips the lookup
  entirely and adds no OSM note — the package should not claim attribution for
  data it did not use, and there is a test for that.
- Added `"SANTO DOMINGO DE LOS TSACHILAS"` alongside the existing
  `"STO DOMINGO DE LOS TSACHILAS"` in `AMBIGUOUS_PROVINCES`. Tabla 19
  abbreviates; OSM spells it out. Without the alias `region_for_provincia()`
  would have fallen through to the generic "Unknown Ecuadorian province"
  message instead of the informative §3.3.1 one.
- Narrowed the `hazard.py` docstring claim "No third-party data ships with
  codeSpectra" to "No third-party **hazard** data" — it was already loose given
  the GeoNames bundle, and is now plainly wrong otherwise.

## Licensing — the part I spent the most care on

OSM is **ODbL v1.0**. The package is MIT. A bundled extract of OSM is a
*Derivative Database* under the ODbL, so it must be distributed under the ODbL,
with attribution and share-alike — it cannot be silently absorbed into an MIT
package. This is materially stricter than the GeoNames precedent (CC BY 4.0,
attribution only), so following that pattern verbatim would have understated the
obligation.

What I did:

- licence, source and attribution recorded in the JSON header, so the notice
  travels with the file itself;
- `OSM_ATTRIBUTION` constant that names OpenStreetMap, the ODbL, *and* states
  explicitly that the table is not covered by codeSpectra's MIT licence;
- every `ProvinceMatch.report()` leads with that notice, and every derived
  region appends it to the site's provenance note — a spectrum built this way is
  a Produced Work, which needs attribution but not share-alike;
- README `## Licence` section rewritten into a table of the two non-MIT bundled
  tables, spelling out that share-alike binds derivatives of the boundary table
  but not spectra produced with it.

I flagged in the response that the alternative — ship the reader, not the data,
as `hazard.py` does for the Palacios layers — is available if the user would
rather not carry share-alike in the repo at all. Bundling was what was asked
for, so bundling is what I did, with the obligation made explicit rather than
buried.

## The real-world discrepancy I found

Cross-checking the polygons against the GeoNames gazetteer surfaced three towns
(`LA VILLEGAS`, `MONTERREY`, `PLAN PILOTO`) that Tabla 19 puts in **Esmeraldas**
but OSM puts in **Santo Domingo de los Tsáchilas**. Confirmed via an Overpass
`is_in` query: they are in cantón **La Concordia**, which changed province after
NEC-SE-DS 2015 was published.

This is a genuine trap, not a data bug: OSM tracks today's administrative map,
while §3.3.1 inherits older groupings. It matters because the two answers differ
in outcome — Esmeraldas is explicitly Sierra (η = 2.48), while Santo Domingo is
placed in no region at all and therefore raises. Raising is the correct
behaviour: it forces the engineer to decide which province the standard meant
instead of silently returning 2.48 or 1.80.

Documented in the module docstring, in every report note, and in the README.
Two tests pin it — one allows the three towns as known exceptions in the
agreement check, one asserts the disagreement still exists so the caveat cannot
rot unnoticed.

`GENERAL FARFAN` was the fourth mismatch: 0.55 km outside Sucumbíos on the
Colombian river border. That is what motivated the 2 km `tolerance_km`, with
`inside=False` recorded on the match so the approximation is visible rather than
hidden. It is the fixture for the tolerance tests.

## Quality gate — PASS

```
$ python -m ruff check src tests
All checks passed!

$ PYTHONPATH=src python -m mypy src
Success: no issues found in 28 source files

$ PYTHONPATH=src python -m pytest
582 passed in 1.06s
```

67 of those are new (`tests/test_nec_provincias.py`), plus 4 rewritten/added in
`tests/test_nec_hazard.py`. Docstring examples in `provincias.py` also pass
under `--doctest-modules`.

One ruff finding during development (RUF012, mutable class attribute in the new
test) was fixed with `ClassVar` rather than suppressed.

`tests/test_nec_hazard.py::test_region_is_not_guessed_from_coordinates` was
renamed and rewritten — its name asserted the old behaviour. It is now
`test_an_explicit_region_overrides_the_derivation`, keeping the original
assertion (costa ≠ sierra) and adding that the default now derives sierra.

## What I deliberately did not do

- No cantonal or parroquial boundaries. §3.3.1 works at province level; going
  finer would multiply the bundle size for nothing.
- No build/vendoring script. The JSON header carries the full Overpass query
  and parameters, which matches how the gazetteer documents itself.
- No change to `nec_site_from_poblacion` — it already derives the region from
  the Tabla 19 provincia, which is *code data* and strictly better than
  geometry. Coordinate derivation is the fallback for when there is no
  poblacion, not a replacement.
