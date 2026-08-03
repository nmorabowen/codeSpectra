# NCh433 Tabla 5.1 added

Tabla 5.1 ("Valores máximos de los factores de modificación de la respuesta", p. 27 of
NCh433.Of1996 Mod.2009 with DS 61-2011) is now transcribed into the NCh433 module, and
`design_spectrum()` / `reduction_factor()` / `C_max()` can read `Ro` and `R` from it — or
check the value you pass against it.

## What went in

New file `src/codeSpectra/codes/nch433/_tables.py`:

- **`StructuralSystem`** — a `str` Enum with one member per row of Tabla 5.1 (16 rows).
  Members carry `.R`, `.Ro` and `.factors` shortcuts.
- **`ResponseFactors`** — a frozen dataclass holding the row: `system`, `family`
  (*Pórticos* / *Muros y sistemas arriostrados* / *Sin clasificar*), `material` (the
  second column as printed), `R`, `Ro`, and the footnotes that apply. It has a
  `report()` that renders with the `ClauseRef` for §5.7.1 / Tabla 5.1, like everything
  else in the library.
- **`TABLE_5_1`** — the dict, keyed by system.
- **`response_factors(system)`** — the lookup. Accepts an enum member or its string;
  an unknown system raises `TableLookupError` listing the valid keys.
- **`resolve_Ro(Ro=None, system=None)`** and **`resolve_R(...)`** — the checking layer.

The table as transcribed:

| Sistema | Material | R | Ro |
| --- | --- | --- | --- |
| Pórticos | Acero: a) Marcos corrientes (OMF) | 4 | 5 |
| Pórticos | Acero: b) Marcos intermedios (IMF) | 5 | 6 |
| Pórticos | Acero: c) Marcos especiales (SMF) | 7 | 11 |
| Pórticos | Acero: d) Marco de vigas enrejadas (STMF) | 6 | 10 |
| Pórticos | Hormigón armado | 7 | 11 |
| Muros y sistemas arriostrados | Acero: a) Marcos concéntricos corrientes (OCBF) | 3 | 5 |
| Muros y sistemas arriostrados | Acero: b) Marcos concéntricos especiales (SCBF) | 5,5 | 8 |
| Muros y sistemas arriostrados | Acero: c) Marcos excéntricos (EBF) | 6 | 10 |
| Muros y sistemas arriostrados | Hormigón armado | 7 | 11 |
| Muros y sistemas arriostrados | Hormigón armado y albañilería confinada — cumple criterio A | 6 | 9 |
| Muros y sistemas arriostrados | Hormigón armado y albañilería confinada — no cumple criterio A | 4 | 4 |
| Muros y sistemas arriostrados | Madera | 5,5 | 7 |
| Muros y sistemas arriostrados | Albañilería confinada | 4 | 4 |
| Muros y sistemas arriostrados | Albañilería armada — huecos llenos / doble chapa | 4 | 4 |
| Muros y sistemas arriostrados | Albañilería armada — rejilla / huecos sin llenar | 3 | 3 |
| Sin clasificar | Cualquier otro tipo de estructuración o material | 2 | — |

All three footnotes are carried as `notes` on the rows they apply to (footnote 1 on the
steel/concrete rows, footnote 2 — the definition of *criterio A* — on the two mixed
wall rows, footnote 3 on the last row).

## The `Ro` check you asked for

`design_spectrum()` and `reduction_factor()` now take an optional `system=` keyword.
`Ro` stays the second positional argument, so existing calls are unaffected.

```python
from codeSpectra.codes.nch433 import NCh433, StructuralSystem, response_factors

site = NCh433(zone="3", soil="D", category="II")

# 1. Read Ro straight from Tabla 5.1
s = site.design_spectrum(T_star=0.8, system=StructuralSystem.PORTICO_HORMIGON_ARMADO)
# label: "NCh433 design spectrum (R*=6.415, T*=0.8 s, Ro=11)"

# 2. Use your own Ro, checked against the tabulated maximum
site.design_spectrum(T_star=0.8, Ro=9.0, system="portico_hormigon_armado")   # fine
site.design_spectrum(T_star=0.8, Ro=12.0, system="portico_hormigon_armado")
# InvalidInput: Ro = 12 exceeds the maximum 11 that NCh433 Tabla 5.1 allows for
# 'portico_hormigon_armado' (Porticos — Hormigon armado). Tabla 5.1 gives maximum
# values; §5.7.2 and §5.7.3 may require a smaller one still.

# 3. Or just inspect the row
print(response_factors("muro_acero_scbf"))   # R = 5.5, Ro = 8, with the clause and footnotes
```

`C_max()` got the same treatment on the `R` side (`site.C_max(system="muro_madera")`),
since Tabla 6.4 is keyed by the same `R` the new table bounds.

Two points where I deliberately refused to invent a number:

- The last row of Tabla 5.1 has no `Ro` — footnote 3 says modal spectral analysis does
  not apply there at all. That cell is `None`, `ResponseFactors.modal_spectral_analysis_permitted`
  is `False`, and `design_spectrum(..., system="otro")` raises with the footnote quoted
  rather than silently substituting `R = 2`.
- The table's values are **maxima**. The error message points at §5.7.2 and §5.7.3
  (mixed systems and mixed directions take the *smaller* factor), because the library
  cannot see your framing and should not imply the tabulated value is automatically
  yours to use.

## Tests

57 new tests in `tests/test_nch433.py` (104 in that file, 562 in the suite, all green).
They assert every printed `R`/`Ro` cell against a literal table transcribed independently
of the source module, check that the enum and the dict stay in sync, that `Ro >= R` on
every row, that the footnotes land on the right rows, that reports are cp1252-safe, and
that the resolve/validate paths behave — including the backwards-compatible bare-`Ro`
call.

`ruff check` clean; `mypy` clean on `src` plus the NCh433 tests (the 10 pre-existing
errors in `test_core.py`, `test_export.py` and `test_nec_hazard.py` are unchanged from
`bd20bec`).

## Note on the transcription

`pdftotext -layout` reflows Tabla 5.1 badly — the vertically-merged "Muros y sistemas
arriostrados" label lands on the *Madera* line and the criterio-A sub-rows drift off
their numbers. I re-extracted the table page with `pdftotext -raw`, which emits one
sub-row per line with its `R` and `Ro` adjacent, and transcribed from that. Worth knowing
if you go after Tabla 6.4 or Tabla 4.2 the same way.

One printing quirk kept as-is: the standard prints SCBF as `5.5` (decimal point) and
Madera as `5,5` (decimal comma) on the same page. Both are 5.5.

## Not done

- No change to Eq. 6-11 (the alternative `R*` for wall-type buildings) — still
  unimplemented and still called out in `report().notes`.
- §5.7.2 / §5.7.3 are documented in the error text but not automated: the library has no
  model of a building's storeys or directions, so picking the governing minimum across
  mixed systems is still yours.
