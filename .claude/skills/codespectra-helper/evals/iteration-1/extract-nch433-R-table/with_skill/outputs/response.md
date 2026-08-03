# NCh433 Tabla 5.1 is in

All 16 rows of Tabla 5.1 (page 27 of the DS 61-2011 PDF) are transcribed into a new
`src/codeSpectra/codes/nch433/_tables.py`, with a lookup and with every method that
consumes `R` or `Ro` now able to take the system instead of a hand-typed number.

## Using it

```python
from codeSpectra.codes.nch433 import NCh433, structural_system

row = structural_system("porticos_acero_smf")
row.R, row.Ro            # (7.0, 11.0)
row.label                # 'Porticos - Acero estructural - c) Marcos especiales (SMF)'
print(row.report().to_text())   # the two values, cited to Tabla 5.1, with footnotes

site = NCh433(zone="2", soil="B", category="II")
site.design_spectrum(T_star=1.0, system="porticos_acero_smf")   # Ro from the table
site.C_max(system="porticos_acero_smf")                         # R from the table
```

`reduction_factor`, `design_spectrum` and `C_max` all keep their old positional
signature, so `design_spectrum(1.0, Ro=11.0)` still works exactly as before.

## The bit you actually asked for: a way to check it

Tabla 5.1 prints **maxima** (§5.7.1 and the table's own caption), so passing a
smaller factor is legitimate and passing a larger one is not. If you give both a
value and a system, the value is used but bounded:

```python
site.reduction_factor(1.0, 8.0, system="porticos_acero_smf")   # fine, 8 <= 11
site.reduction_factor(1.0, 12.0, system="porticos_acero_smf")  # InvalidInput
```

Passing neither now raises rather than silently defaulting.

## Two things in the table worth knowing about

**The last row has no `Ro`.** "Cualquier tipo de estructuracion o material que no
pueda ser clasificado en alguna de las categorias anteriores" is given `R = 2` and a
dash for `Ro`, because footnote 3 forbids modal spectral analysis for it outright. I
stored that cell as `None` rather than inventing a number, so:

```python
site.design_spectrum(1.0, system="otro")
# InvalidInput: NCh433 Tabla 5.1 establishes no Ro for 'otro' (...). No procede el
# uso del analisis modal espectral para este tipo de estructuracion o material. ...
# Use the static method with R = 2 instead.
```

The static path still works — `structural_system("otro").R` is 2.0 and
`C_max(system="otro")` is fine.

**`R` and `Ro` are not interchangeable.** `R` feeds the static method (Eq. 6-2) and
`Ro` feeds Eq. 6-10. Tabla 6.4's `C_max` is indexed by `R`, not `Ro`, and it was
already being passed one by hand too — that's why `C_max` got the same treatment.

The three printed footnotes are carried on each row (`row.footnotes`,
`row.notes()`) and land in `row.report()`, including the Criterio A definition
("the reinforced-concrete walls must take at least 50% of the storey shear at
every storey") that distinguishes the `6/9` row from the `4/4` one.

## Keys

Sixteen ASCII slugs, in printed order:

| Key | R | Ro |
| --- | --- | --- |
| `porticos_acero_omf` | 4 | 5 |
| `porticos_acero_imf` | 5 | 6 |
| `porticos_acero_smf` | 7 | 11 |
| `porticos_acero_stmf` | 6 | 10 |
| `porticos_hormigon_armado` | 7 | 11 |
| `muros_acero_ocbf` | 3 | 5 |
| `muros_acero_scbf` | 5.5 | 8 |
| `muros_acero_ebf` | 6 | 10 |
| `muros_hormigon_armado` | 7 | 11 |
| `muros_hormigon_albanileria_criterio_a` | 6 | 9 |
| `muros_hormigon_albanileria_sin_criterio_a` | 4 | 4 |
| `muros_madera` | 5.5 | 7 |
| `albanileria_confinada` | 4 | 4 |
| `albanileria_armada_bloques_llenos` | 4 | 4 |
| `albanileria_armada_rejilla_o_bloques_sin_llenar` | 3 | 3 |
| `otro` | 2 | — |

Lookup is case- and whitespace-insensitive; an unknown key raises `TableLookupError`
listing all sixteen.

## How I checked the extraction

I pulled page 27 twice, with `pdftotext -table` and with `-layout`, and the two
agree row-for-row — including the group-label placement, which is vertically
centred in the merged cell and so lands mid-block in both extractions (`Porticos`
next to the IMF row, `Muros y sistemas arriostrados` next to `Madera`). That
placement is what fixes the `Acero estructural` sub-block boundary: OMF/IMF/SMF/STMF
belong to Porticos, OCBF/SCBF/EBF to Muros. The PDF prints `5.5` for SCBF and `5,5`
for Madera; both are 5.5.

Descriptions are transcribed without accents, matching the existing NCh433 module's
convention, so `report().to_text()` survives a cp1252 console — asserted per row.

## Tests

109 new tests in `tests/test_nch433.py`, covering every cell against a literal
transcription of the printed table, row count and printed order, the distinct value
sets, footnote assignment, lookup behaviour, the no-`Ro` row, the maxima
enforcement, and cp1252-safety of every row's report.

Quality gate is clean: `pytest` (614 passed), `ruff check .`, `mypy src/codeSpectra`
(strict), and the 3.10 AST parse check.

## Not done

Eq. 6-11, the alternative `R*` for wall buildings, is still unimplemented — that's
unchanged and still flagged in the site report's notes.
