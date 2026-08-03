Done — `region=` is now derived from the coordinates, and the province
boundaries ship with the package. One licensing consequence you need to sign
off on, which I'll put first because it changes what your wheel is.

## The licence, up front

OpenStreetMap is **ODbL 1.0**, not the CC BY that the GeoNames gazetteer uses.
ODbL clearly permits redistribution, so bundling it is legitimate — unlike the
Palacios PSHA layers, which have no licence at all. But ODbL is **share-alike**,
and that has a real effect:

- The bundled polygons are a *Derived Database*. They go out under the ODbL, so
  **codeSpectra's wheel is no longer uniformly MIT** — it is MIT code plus one
  ODbL data file. I've recorded that in `LICENSE` under "Third-party data", in
  `codes/nec/tables/LICENSE-OpenStreetMap.txt`, and in the README.
- **Your users' calculations are not infected.** A spectrum, a base shear, a
  design report computed *with* the boundaries is a *Produced Work* under
  ODbL §4.5: it needs the attribution notice, and nothing more. Share-alike
  bites only if someone redistributes the boundary database itself. I've said
  this explicitly in the notice file, because an engineer who sees "share-alike"
  on a seismic library will otherwise assume their calc package is affected.
- Attribution now travels automatically: `OSM_ATTRIBUTION` is in the derived
  region's provenance note and in every `ProvinceMatch.report()`.

This is an established pattern (`timezonefinder` ships MIT code with ODbL
OSM-derived polygons), so I went ahead — but it is your call, and it is a
one-file swap if you'd rather keep the wheel uniformly permissive:
`ProvinceBoundaries.from_file()` takes any admin-1 GeoJSON, so Natural Earth
(public domain) or geoBoundaries (CC BY 4.0, same class as GeoNames) would drop
straight in with no share-alike component.

## What it does

```python
from codeSpectra.codes.nec import provincia_at, region_at

region_at(lat=-2.1900, lon=-79.8875)          # 'costa'   (Guayaquil)
m = provincia_at(lat=-0.1807, lon=-78.4678)
m.provincia, m.region, m.margin_km            # 'Pichincha', 'sierra', 30.7

site = nec_site_from_hazard(est, soil="D")    # region= no longer needed
```

The important point is that this is not a new rule — it's the code's own rule
applied to a position. **NEC-SE-DS §3.3.1 groups *provinces*** into Costa,
Sierra and Oriente; all the boundaries add is which province a point is in. So
the derivation reuses the existing `region_for_provincia`, and the report says
which half came from the standard and which did not.

## The bug underneath the annoyance

`nec_site_from_hazard` didn't just *require* `region` — it defaulted to
`region="sierra"`. Any caller who omitted it silently got η = 2.48 applied to a
Guayaquil site: 38% too much short-period demand, with nothing in the report
saying so. That default is gone; `region=None` now means "derive it", and there
is no path that quietly picks a region for you.

## Four things it refuses to do

- **Outside Ecuador** → `ProvinceNotFound`, naming the nearest province and its
  distance. A sign-flipped longitude reads as "nearest province 8,700 km away"
  instead of returning something plausible.
- **Santo Domingo de los Tsáchilas** → still raises. §3.3.1 never placed it
  (created 2007, on the Costa–Sierra transition), so neither does the lookup.
  Supply `region=` for those sites. 123 of the 8,207 mainland grid points I
  audited land there, so it is not an edge case you can ignore.
- **On a line between two regions** → `AmbiguousProvince`. Shared borders are
  simplified once per province, and Douglas-Peucker isn't invariant to where a
  ring is cut, so the two sides can disagree by a couple of hundred metres. In
  that band the geometry cannot settle a 1.80 / 2.48 / 2.60 difference, so the
  lookup answers only when every candidate province is in the *same* region,
  and refuses when they are not. Walking the entire Guayas border, 3 of 427
  sample points hit that refusal; the rest resolved cleanly.
- **Near a line, but resolvable** → answers, and says so. `margin_km` reports
  how far inside the province the site sits, and the note shouts when a
  differently-regioned neighbour is within 5 km.

One divergence worth knowing: Tabla 19 still lists `ZONA NO DELIMITADA` rows,
but OSM's present-day boundaries assign that territory to the provinces that
absorbed it, so a coordinate there resolves normally while the *name* lookup
still raises. That is the two sources disagreeing about a 2015 change, not a
bug — I left both behaviours as they are and documented it.

## Data and reproducibility

`scripts/build_provincias_osm.py` is the complete derivation: Overpass query
for `admin_level=4` inside `ISO3166-1=EC`, member ways assembled into closed
rings, Douglas-Peucker at 0.002° (~220 m, measured worst deviation 223 m),
coordinates rounded to 4 dp. 24 provinces, 228 kB, and it validates before it
writes — 24 rows, unique ISO codes, closed rings, a non-ASCII audit, exactly
one province unplaced by §3.3.1, and 12 hand-checked cities through the
library's own reader.

That validation caught a real failure: `overpass.osm.ch` carries **Switzerland
only** and answered 200 OK with an empty element list for 16 of the 24
provinces. Valid JSON, zero data — it would have shipped 16 provinces with no
geometry. The mirror list is now restricted to world-coverage instances and
`_overpass()` treats an empty result as an error.

## Gate

`pytest` 583 passed · `ruff` clean · `mypy src/codeSpectra` clean · 3.10 AST
parse clean · wheel contains `provincias_osm.json` **and**
`LICENSE-OpenStreetMap.txt` (I widened the `package-data` glob to `*.txt` for
the notice — worth checking, since a missed glob is invisible until after
install).

71 new test cases in `tests/test_nec_provincias.py`, across the usual six
categories (18 cities parametrised twice, plus data integrity, refusals,
provenance, encoding and custom-layer handling).
I also renamed `test_region_is_not_guessed_from_coordinates` to
`test_an_explicit_region_overrides_the_derivation` — the old name encoded the
policy this change reverses, though its assertion still holds.
