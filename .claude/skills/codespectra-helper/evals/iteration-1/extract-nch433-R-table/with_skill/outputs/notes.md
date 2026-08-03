# Notes — NCh433 Tabla 5.1 (R and Ro)

## 0. Worktree base correction (do this first, it was nearly a silent failure)

Two things about the environment mattered:

1. My assigned worktree (`agent-adc98ba290d934946`) was checked out at `c00751e`,
   which is the *pre-library* state — a single flat `codeSpectra/designSpectra.py`
   with a class-based NCh433 and no tests. The actual library the task describes
   (`src/codeSpectra/codes/nch433/`, `design_spectrum()`, the skill itself) lives at
   `52cc4d5` on `claude/asce-nec-spectrum-library-5739fa`. I `git reset --hard
   52cc4d5` in my own worktree so the patch is against the right base. Without this
   the whole task would have been done against the wrong file.
2. The `codeSpectra` editable install
   (`__editable__.codespectra-0.2.0.pth`) points at a *different* worktree
   (`asce-nec-spectrum-library-5739fa/src`). My first `pytest` run therefore passed
   while testing someone else's code. It is a simple path `.pth`, not a meta-path
   finder, so `PYTHONPATH=<my worktree>/src` takes precedence; every gate run below
   sets it. Verified with `codeSpectra.__file__`.

## 1. Extraction

Source: `C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos\Chile\
NCh433-Of-1996-Mod-2009-DS-61-2011.pdf`.

Located the table by probing pages 1..140 for `Tabla\s*5\.1`; hits at pages 3 (index),
34 (§5.7 cross-reference), 37 (the table itself, printed page 27) and 56 (Anexo B).

Per the skill's extraction reference, `pdftotext` here is Xpdf 4.00. I extracted page
37 three ways — `-table`, `-layout`, `-raw` — and used the first two.

The table is small and clean enough to transcribe by hand (16 rows), so no positional
parser was needed. The risk here is not wrapped cells but **row grouping**: the
*Sistema estructural* column is a merged cell whose label is vertically centred, so
in a linearised extraction it lands in the middle of its block rather than at the
top. Both extractions place `Pórticos` beside the IMF row and `Muros y sistemas
arriostrados` beside `Madera`, which is what resolves the ambiguity of the two
separate `Acero estructural` sub-blocks:

- Pórticos: OMF, IMF, SMF, STMF, Hormigón armado (5 material rows; label at row 3)
- Muros y sistemas arriostrados: OCBF, SCBF, EBF, Hormigón armado, Hormigón armado y
  albañilería confinada (2 variants), Madera, Albañilería confinada, Albañilería
  armada (2 variants) (11 rows; label at row 6)

Both midpoints check out, and the semantics agree (moment frames vs braced
frames/walls).

## 2. Validation of the extraction

| Check | Result |
| --- | --- |
| `-table` vs `-layout` cross-check | agree row-for-row on all 16 rows and both value columns |
| Row count | 16 rows: 5 + 10 + 1 catch-all, matching the printed block structure |
| Distinct `R` set | {2, 3, 4, 5, 5.5, 6, 7} — asserted in a test |
| Distinct `Ro` set | {3, 4, 5, 6, 7, 8, 9, 10, 11, None} — asserted in a test |
| No empty text fields | asserted per row |
| Non-ASCII audit | keys and both text columns asserted `isascii()`; no font corruption of the kind seen in the NEC PDF (this PDF renders á/ó/ñ correctly) |
| Key uniqueness | 16 distinct keys, asserted via row count vs. the literal test table |
| Ordering | `structural_system_keys()` asserted equal to printed order |
| Spot check | SMF 7/11, SCBF 5.5/8, EBF 6/10, Madera 5.5/7, Albañilería confinada 4/4, catch-all 2/— |

Decimal-separator inconsistency in the source: SCBF prints `5.5`, Madera prints
`5,5`. Both are five-and-a-half; recorded as `5.5`.

Descriptions are transcribed **without accents**, matching the existing
`nch433.py` convention (`Aceleracion efectiva maxima` etc.). cp1252 would have
tolerated accents, but mixing conventions inside one report is worse; asserted
cp1252-safe per row regardless.

## 3. What I built

New `src/codeSpectra/codes/nch433/_tables.py` (following the `asce7/_tables_7_16.py`
and `nec/_tables.py` convention):

- `StructuralSystem` — frozen slotted dataclass: `key`, `system`, `material`, `R`,
  `Ro`, `footnotes`; plus `label`, `modal_spectral_permitted`, `ref`, `require_Ro()`,
  `notes()`, `report()`.
- `STRUCTURAL_SYSTEMS` — the 16 rows in printed order.
- `TABLA_5_1_FOOTNOTES` — the three printed footnotes, including the Criterio A
  definition that distinguishes the 6/9 row from the 4/4 one.
- `structural_system(key_or_row)` — case/whitespace-insensitive lookup, passthrough
  for an instance, `TableLookupError` listing all keys otherwise.
- `check_R` / `check_Ro` — resolve a factor from an explicit value, a system, or
  both. `InterpolatedTable` was deliberately *not* used: Tabla 5.1 is a categorical
  row list with two independent value columns, not a row × column interpolation.

`STANDARD`, `EDITION` and the `_ref` helper moved into `_tables.py` (re-exported from
`nch433.py`) to avoid duplicating them across the two modules.

Wiring in `nch433.py`: `reduction_factor`, `design_spectrum` and `C_max` all gained a
keyword-only `system=`, with the explicit factor now optional. Old positional calls
are unchanged. Added `NCh433.structural_system()` staticmethod for discoverability
from the already-top-level-exported class, and a note in `NCh433.report()` pointing
at Tabla 5.1.

### Design decisions worth flagging

- **The catch-all row's `Ro` is `None`, not a number.** The PDF prints a dash and
  footnote 3 says modal spectral analysis does not apply, so no `Ro` is established.
  Per the prime directive this raises `InvalidInput` (quoting the footnote and
  redirecting to the static method with `R = 2`) rather than returning something
  plausible. `SiteSpecificRequired` would have been the wrong exception — nothing is
  deferred to a study; the analysis method is simply prohibited.
- **The table's values are maxima**, per §5.7.1 and the caption *"Valores máximos"*.
  So a smaller user-supplied factor is accepted and a larger one raises. This is the
  "way to check it" the request was actually about.
- **`C_max` got the same treatment** because Tabla 6.4 is indexed by `R` — the
  static-method factor from the same Tabla 5.1 — and was equally unchecked.
- **Not added to the top-level `codeSpectra.__getattr__`.** `StructuralSystem` and
  `structural_system` are NCh433-specific but generically named; in a four-standard
  library they would read as universal. They are exported from
  `codeSpectra.codes.nch433`, and reachable as `NCh433.structural_system(...)`.

## 4. Tests

109 new tests in `tests/test_nch433.py`, in the skill's categories:

- **Table integrity** — a module-level `TABLA_5_1` literal transcribing the printed
  table, parametrised cell-by-cell; row count; printed order; distinct value sets;
  key hygiene; text-column population and ASCII-ness; footnote assignment (footnote 1
  asserted to apply to exactly the steel and reinforced-concrete rows).
- **Code rules** — maxima enforced in both directions; missing both value and system
  raises; non-positive still raises; the no-`Ro` row raises on every modal path while
  the static path still works.
- **Lookup** — case/whitespace insensitivity, instance passthrough, unknown key
  raising and naming the table, immutability.
- **Integration** — `design_spectrum(system=)` and `C_max(system=)` numerically equal
  the explicit-factor calls.
- **Encoding** — every row's `report().to_text().encode("cp1252")`.
- **Citations** — every report item carries the Tabla 5.1 `ClauseRef`; the no-`Ro`
  row renders a blank rather than a number.

## 5. Quality gate

Run with `PYTHONPATH` pinned to this worktree's `src`:

```
python -m pytest -o addopts=""     ->  614 passed   (505 before; +109)
python -m ruff check .             ->  All checks passed!
python -m mypy src/codeSpectra     ->  Success: no issues found in 28 source files
python -c "ast.parse(..., feature_version=(3,10))" over src/**/*.py  ->  ok
python -m pytest --doctest-modules src/codeSpectra/codes/nch433  ->  3 passed
```

Two ruff findings were fixed during the run: `RUF043` on two unescaped `match="Tabla
5.1"` regexes. No new `tables/*.json` was added, so the wheel `package-data` check
was not applicable.

The doctest run caught a wrong expected value I had written from memory
(`reduction_factor(1.0, system="porticos_acero_smf")` is 9.271, not 8.462) — worth
noting since the suite does not run doctests by default.

## 6. Left alone

Eq. 6-11 (alternative `R*` for wall buildings) remains unimplemented; the existing
note in `NCh433.report()` still says so, and README's "Not implemented" list is
unchanged on that point. README updated only for the Tabla 5.1 coverage line and the
capability summary row.

Nothing was committed.
