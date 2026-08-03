# Notes — deriving the NEC region from coordinates

Worktree: `C:\Users\nmora\Documents\GitHub\codeSpectra\.claude\worktrees\agent-ae4368962ae096a7e`
Base: `198bbad` (reset as instructed; `src/codeSpectra/codes/` and the skill file both confirmed present).
All Python run with `PYTHONPATH=src` so the local checkout is exercised, not the editable install elsewhere on the machine.

---

## 1. Decision: was new data actually needed?

The request assumed boundaries were required. They were not obviously so — the
repo already ships everything needed for a cheaper derivation:

- `Tabla19` (515 rows) names the `provincia` of every entry;
- `Gazetteer.geonames()` places 453 of them;
- `region_for_provincia()` already implements the §3.3.1 grouping.

So `Tabla19.nearest(lat, lon, gazetteer).poblacion.provincia → region` is a
zero-new-data, zero-new-licence answer. Before adding a 446 KiB share-alike
dataset I measured whether it buys anything.

**Method.** Downloaded geoBoundaries ECU ADM1 as neutral ground truth (CC0,
independent of OSM), built point-in-polygon, and compared against nearest-town
on a 0.02° grid over Ecuador (50,468 land cells).

**Result.**

| | |
|---|---|
| comparable cells | 49,423 |
| agree | 46,304 (**93.69%**) |
| disagree | 3,119 (**6.31%**) |

Disagreement by direction: oriente→sierra 1,577; sierra→costa 1,017;
costa→sierra 385; sierra→oriente 140. Median distance to the nearest listed
town at a disagreeing cell: 25.2 km (p90 40.1, max 57.5).

The sierra→costa class is the dangerous one: `eta` 2.48 → 1.80, a 27%
under-estimate of the plateau ordinate. 6.3% of land area with a failure mode
that magnitude is not acceptable as a silent default.

**Conclusion: real polygons are justified.** Recorded here because the opposite
conclusion was plausible going in, and the measurement is what settled it.

## 2. Decision: which source, and the licence question

This is the §9 question in the skill ("ship the reader, not the data, unless
the licence clearly permits redistribution"). Two candidates:

| | OSM (Overpass `admin_level=4`) | geoBoundaries gbOpen ECU ADM1 |
|---|---|---|
| Licence | **ODbL 1.0** — attribution + **share-alike** | **CC0 1.0** — no conditions |
| Provenance | unambiguous, well documented | `boundarySource: "geoBoundaries, Wikimedia Commons"`, `licenseDetail: nan`, `licenseSource: commons.wikimedia.org/wiki/File` |
| Vintage | current | `boundaryYearRepresented: 2011` |
| Data defects found | none | **Cotopaxi and Chimborazo both tagged `EC-H`** (Cotopaxi should be `EC-X`) |

§9's test is *"unless the licence clearly permits redistribution"*. ODbL
**does** clearly permit it — that is the real difference from the Palacios
PSHA map, which has no licence at all and so is all-rights-reserved. So a
blanket refusal to bundle OSM would have been hiding behind the rule rather
than applying it.

But permitting redistribution is not the same as costing nothing. ODbL is
share-alike and codeSpectra is MIT, so bundling changes the package's
licensing position. I judged the honest move to be: bundle it (the user asked,
the licence allows it, and OSM is the better data), satisfy the ODbL
obligations properly, and put the cost in front of the user explicitly with
the CC0 alternative named and costed.

Three obligations, all wired up:

1. **The derived database stays ODbL.** `provincias_osm.json` carries its own
   `licence`/`attribution`/`derived` header naming the simplification, since
   simplifying is itself creating a Derivative Database.
2. **`LICENSE` states the split.** New third-party section naming the file,
   the obligation on redistributors, and the exit (delete the file, pass
   `region=`; nothing else depends on it). Verified it reaches the wheel's
   `dist-info/licenses/LICENSE` and the ODbL text appears in METADATA.
3. **Produced works carry attribution.** `OSM_ATTRIBUTION` is emitted
   automatically inside `ProvinceLocation.provenance_note()`, so any report
   whose region was derived carries it without the engineer remembering.

Also updated the skill's §9 table and added the "permits redistribution ≠ costs
nothing" rule, plus a note not to make a habit of share-alike datasets.

## 3. Extraction (skill §7 discipline applied to geometry)

`scripts/extract_province_boundaries.py`. Network only under `--download`.

Traps hit and handled:

- **Foreign relation leaked in.** The Overpass `area["ISO3166-1"="EC"]` query
  returned 25 relations, not 24 — Nariño (`CO-NAR`) shares a border way with
  Carchi. Filtered on `ISO3166-2 ^EC-`; the rejects are logged, and a test
  asserts Nariño stayed out. This is precisely the §7 "assert the row count
  and the leftovers" lesson in a geometric register.
- **Ring assembly.** Overpass returns member ways in relation order, not
  traversal order, some digitised backwards. Chained by endpoint with
  reversal; a chain that fails to close raises rather than shipping an open
  ring.
- **Accented names.** OSM has `Manabí`, `Cañar`, `Sucumbíos`, `Galápagos`.
  Folded to ASCII at extraction (reports print on cp1252 consoles, skill §5),
  with a test asserting every shipped name is ASCII.
- **Name mismatch across sources.** Tabla 19 writes `STO. DOMINGO DE LOS
  TSACHILAS`; OSM spells it out. Found that `region_for_provincia` returned
  "Unknown Ecuadorian province" for the spelled-out form instead of the
  explanatory §3.3.1 message. Both spellings now map to the real message.
  This was a latent bug, not something the new feature introduced.

Simplification cost, measured rather than assumed: 158,543 → 22,009 vertices
at 100 m Douglas-Peucker. Compared full-resolution against simplified on an
8,137-cell grid — 7 cells change province (0.086%), all on boundaries, and
**none of them change region** (Pastaza/Napo and Orellana/Sucumbíos are both
oriente, Tungurahua/Cotopaxi both sierra).

Coordinates rounded to 5 dp (~1.1 m, below the tolerance). File is 446 KiB.
Re-running the extractor reproduces it byte-identically.

**Independent cross-check.** Every Tabla 19 town the GeoNames gazetteer can
place should fall inside the province Tabla 19 names for it — three unrelated
datasets agreeing. 449/453 (99.1%). The four exceptions are all explained:

- Monterrey, La Villegas, Plan Piloto (3): parroquias of La Concordia,
  transferred Esmeraldas → Santo Domingo in 2013. Tabla 19 predates the
  transfer, OSM postdates it. Neither is wrong. **This one has teeth**: §3.3.1
  gives Esmeraldas the Sierra eta and does not assign Santo Domingo at all.
  Pinned in `test_la_concordia_is_a_known_gazetteer_disagreement` so it cannot
  drift into being an unexplained mismatch.
- General Farfán, Sucumbíos (1): the GeoNames coordinate sits across the San
  Miguel river in Colombia. Outside every polygon at *full* resolution too, so
  not a simplification artifact.

## 4. Design decisions in the library

- **`_geo.py`** — `haversine_km` and `point_to_polyline_km` were duplicated
  between `poblaciones.py` and `hazard.py` (plus two copies of the Earth
  radius). The new module would have made it a third copy, so they moved to a
  shared private module.
- **`boundary_km` measures to the nearest *other* province**, not to the
  containing polygon's outline. First implementation used the outline and
  flagged Esmeraldas city as 3.5 km from a "boundary" — that boundary was the
  Pacific. The ocean cannot change which region applies.
- **`snap_km=2.0`** absorbs offshore sites, border-river coordinates and the
  simplification tolerance. `find()` returns None, `locate()` raises with the
  distance; both are explicit rather than silently extrapolating.
- **`_eta()` reads `Region(name).eta`** rather than a local dict, so the
  warning text cannot drift from the code values.
- **Raises rather than guesses**, per the prime directive: Santo Domingo
  (§3.3.1 is silent), and anywhere outside Ecuador. The latter is a real
  behaviour improvement — the old signature returned a `"sierra"` site for a
  coordinate in Lima.

Test expectation corrected mid-run (skill §11): I asserted (0, −78.5) was in
Napo/oriente. It is just north of Quito, in Pichincha — the code was right,
the expectation was wrong.

`test_region_is_not_guessed_from_coordinates` in `test_nec_hazard.py` encoded
the old contract in its name and docstring. Deliberately rewritten rather than
left passing-but-misleading.

## 5. Quality gate

| Check | Result |
|---|---|
| `python -m pytest -q` | **582 passed** (was 513; 69 new) |
| `python -m ruff check .` | clean |
| `python -m mypy src/codeSpectra` | clean, `--strict`, 29 files |
| 3.10 AST parse (`feature_version=(3,10)`) | clean — src, tests and scripts |
| `pytest --doctest-modules provincias.py` | passes |
| wheel package-data | `provincias_osm.json` present, 456,882 B; `LICENSE` present with the ODbL section |
| extractor reproducibility | re-run produces an identical file |

Build artifacts (`dist/`, `build/`, `*.egg-info`) removed; `git status` shows
only intended changes. Nothing committed.

## 6. Files

Modified: `LICENSE`, `README.md`, `.claude/skills/codespectra-helper/SKILL.md`,
`src/codeSpectra/codes/nec/{__init__,hazard,poblaciones}.py`,
`tests/test_nec_hazard.py`.

New: `src/codeSpectra/codes/nec/{_geo,provincias}.py`,
`src/codeSpectra/codes/nec/tables/provincias_osm.json`,
`scripts/extract_province_boundaries.py`, `tests/test_nec_provincias.py`.

## 7. Left for the user

The ODbL-vs-CC0 choice is flagged in the response rather than decided
unilaterally, because it changes the licence terms of the distributed package
and that is the maintainer's call, not mine. The swap is one extractor run
against a different source if they prefer CC0 — at the cost of the buggy ISO
codes, the 2011 vintage and the missing La Concordia transfer.
