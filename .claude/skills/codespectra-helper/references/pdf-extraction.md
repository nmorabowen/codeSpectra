# Extracting design-code tables from PDFs

Everything here was learned by getting it wrong on the ASCE 7 and NEC-SE-DS
PDFs. The failures are all *silent* — the extraction succeeds, produces
plausible output, and quietly loses or corrupts rows. That is why the
validation section is the important half of this document.

## Contents

1. [Which tool](#1-which-tool)
2. [Finding the pages](#2-finding-the-pages)
3. [Simple tables](#3-simple-tables)
4. [Multi-page tables with wrapped cells](#4-multi-page-tables-with-wrapped-cells)
5. [Character corruption](#5-character-corruption)
6. [Validation — the part that catches the real bugs](#6-validation)
7. [Storing the result](#7-storing-the-result)

---

## 1. Which tool

`pdftotext` on this machine is **Xpdf 4.00, not Poppler**. Consequences:

- No `-bbox-layout`, so no word-position XML. Don't plan around it.
- `-table` exists and is the best option for tabular pages.
- `-layout` preserves physical layout and is right for prose and for
  cross-checking.
- `-enc UTF-8` works correctly; if output *looks* mangled in the terminal,
  suspect the console, not the file. Check the bytes:

```bash
python -c "print(b'\xc3\x91' in open('out.txt','rb').read())"   # proper UTF-8 enye
```

`pdfplumber` is not installed. If a table truly defeats `-table`, ask before
adding a dependency.

## 2. Finding the pages

Don't guess from the printed page numbers — PDFs have front matter. Probe:

```bash
for p in $(seq 90 130); do
  n=$(pdftotext -table -enc UTF-8 -f $p -l $p "$PDF" - 2>/dev/null \
      | grep -cE "0\.(15|25|30|35|40|50)$")
  [ "$n" -gt 0 ] && echo "page $p: $n candidate rows"
done
```

Probing by *what the data looks like* (a value column pattern) is more robust
than probing by caption text, which often appears in the table of contents too.

## 3. Simple tables

A small coefficient table (ASCE Table 11.4-1) comes out clean with `-table`
and can be transcribed by hand into an `InterpolatedTable`. Still verify
every cell against the PDF, and still cross-check against a raw extraction —
see §5 of the parent skill for why.

## 4. Multi-page tables with wrapped cells

This is where things break. NEC Tabla 19 is 515 rows over 19 pages with cells
that wrap across lines.

**Locate the value column by regex at end-of-line, not by header offset.**

```python
Z_RE = re.compile(r"\s(0\.\d{2})\s*$")
```

The header's column positions and the data rows' column positions can differ
on the same page. NEC Tabla 19 page 98 pads `Z` far to the right of where its
data rows place it; slicing `line[header_z_index:]` returned empty for every
row, so nothing matched and the entire page was silently dropped — 21 towns
including Cuenca. Use the header only for the *text* columns, and find the
value column by pattern.

**Handle wraps in both directions.**

```
                       CHICAN (GUILLERMO        ← leading wrap
PAUTE                  ORTEGA)     PAUTE  AZUAY  0.25
```

```
JUAN DE VELASCO                    COLTA  CHIMBORAZO  0.40
                       JUAN DE VELASCO               ← trailing wrap
```

Buffer non-value lines and prepend them to the next value row; at page end,
attach any remainder **only to fields that are still empty**. Without that
last condition the table caption gets absorbed as a data row.

**Stop at the caption.** `if "Tabla 19" in line: break`.

## 5. Character corruption

Embedded fonts sometimes carry a broken encoding for accented capitals. In
the NEC PDF:

| Extracted | Actual | Evidence |
|---|---|---|
| `U+00D0` Ð | `Ñ` | `BAÐOS DE AGUA SANTA` → Baños de Agua Santa |
| `U+250C` ┌ | `Ú` | `SUC┌A`, whose cantón column reads `SUCUA` |
| `U+00CB` Ë | `Ó` | `SIMËN BOLIVAR` → Simón Bolívar |

Note the same PDF renders `Ñ` correctly 24 times elsewhere — the corruption is
per-font, not per-document, so a spot check will miss it.

**Audit every non-ASCII character** and confirm each is one you expect:

```python
chars = Counter(ch for row in rows for f in FIELDS for ch in row[f]
                if not ch.isascii())
```

Only repair a glyph when context proves the substitution — the same word
spelled correctly in a neighbouring column, or an unmistakable place name.
If it is ambiguous, leave it and flag it.

## 6. Validation

Run all of these before shipping a table. Each one has caught a real fault.

| Check | Catches |
|---|---|
| Row count vs. an independent count of value-bearing lines | dropped pages |
| Set of distinct values vs. the standard's own discrete set | column desync, stray captures |
| No empty required fields | wrap-handling bugs |
| No unconsumed parse buffer at page end | truncated rows |
| Non-ASCII character audit | font corruption |
| Hand-checked spot rows (10–15 well-known ones) | everything else |
| Key uniqueness — is `(name, ...)` actually unique? | silently collapsing distinct rows |

That last one is subtle and worth expanding. NEC Tabla 19 has 30 duplicated
`poblacion` names; 24 span provinces and 20 of those carry a *different* `Z`.
A dict keyed on name alone silently keeps one and drops the rest, which for a
design library means returning the wrong design value. Check uniqueness of
whatever key you intend to use, and if the key is not unique, either widen it
or omit the ambiguous entries — never pick arbitrarily.

Even `(name, province)` was not unique here: `PUEBLO NUEVO` / Guayas exists
in two cantones with `Z` of 0.40 and 0.50. It is omitted from the derived
gazetteer for exactly that reason.

## 7. Storing the result

- Small coefficient tables → module-level `InterpolatedTable` literals, so
  the values sit next to their `ClauseRef`.
- Large row tables → `codes/<family>/tables/<name>.json` with a `source` and
  `note` field recording where it came from and any repairs applied.
- Add the glob to `[tool.setuptools.package-data]` and **verify it reaches a
  built wheel** — see §10 of the parent skill.
