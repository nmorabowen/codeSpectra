# Notes — ASCE 7-22 Fa/Fv request

## Summary

The request was to add Fa/Fv site-coefficient tables to `asce7_22.py` so 7-22
takes `Ss`/`S1` like 7-16 does. I declined the literal request — ASCE 7-22
deleted those tables — and instead made the module's routing explicit and
pinned the deletion with a test.

## Environment wrinkle (worth flagging)

My assigned worktree `.claude/worktrees/agent-ac00e0b1b52484458` was checked
out at `main` (c00751e), which predates the library rewrite: it contains only
`codeSpectra/designSpectra.py` (an early NCh433 class) and no `codes/asce7/`
at all. The library the user is describing lives on branch
`claude/asce-nec-spectrum-library-5739fa` (52cc4d5), checked out in a sibling
worktree.

I did **not** touch the sibling worktree or that branch. I hard-reset my own
disposable agent branch to 52cc4d5 so I had the real code to work against.
The patch therefore applies cleanly on top of 52cc4d5. Nothing was committed.

One consequence: the editable install resolves `codeSpectra` to the *sibling*
worktree's `src/`, so all test/lint runs were done with
`PYTHONPATH=<my worktree>/src` to make sure I was exercising my own edits.
Confirmed via `codeSpectra.__file__` before running.

## Verification against the standard (skill §1: never invent a code value)

Source: `C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos\ASCE\ASCE 7-22.pdf`,
extracted with `pdftotext -layout` (Xpdf 4.00, per skill §7).

Findings:

- **§11.4.3** reads verbatim: "Risk-targeted maximum considered earthquake
  (MCE_R) spectral response acceleration parameters S_S, S_1, S_MS, and S_M1
  shall be obtained from the USGS Seismic Design Geodatabase for the
  applicable site class." No Fa/Fv, no product form.
- **`grep -c "Table 11\.4"` over the full extracted text returns 0.** Neither
  Table 11.4-1 nor 11.4-2 exists anywhere in 7-22, body or commentary.
- The 7-16 §11.4.4 "Site Coefficients" subsection is absent from the 7-22
  section list; 7-22 goes §11.4.3 -> §11.4.4 "Design Spectral Acceleration
  Parameters" (Eqs. 11.4-1/11.4-2, the 2/3 factor) directly.
- **Commentary C11.4.3** states the intent explicitly: "Seismic parameters
  S_MS and S_M1 (and S_DS and S_D1) incorporate site effects, eliminating the
  need for the tables of site factors Fa and Fv of ASCE 7-16."
- C11.4.3 also confirms Chapter 22 prints **S_MS and S_M1** maps for default
  site conditions (not Ss/S1), and the User Notes give the retrieval point as
  the ASCE 7 Hazard Tool, https://asce7hazardtool.online/.

Only surviving `Fa` in 7-22 is in **§12.14.8.1**, the simplified alternative
procedure: `SDS = (2/3) Fa Ss` with "Fa permitted to be taken as 1.0 for rock
sites, 1.4 for soil sites". That is a two-value rule of thumb inside a
different, restricted procedure — not a site-coefficient table, and not what
was asked for. Not implemented; out of scope here.

So the request rests on a false premise, and satisfying it would have meant
fabricating a 7-22 table from 7-16 values. Per skill §1 the correct outcome
was to say so.

## What I changed instead

`src/codeSpectra/codes/asce7/asce7_22.py`

1. Module docstring: added the C11.4.3 rationale, an explicit "do not add
   Fa/Fv here" with the reason (a 7-16 coefficient on 7-22 values is neither
   edition), and routing for the three situations someone holding Ss/S1 can
   actually be in — has a lat/long (use the hazard tool, get SMS/SM1 or MPRS
   for the same query), reading Chapter 22 print maps (those already plot
   SMS/SM1), or carrying 7-16 values (use `ASCE7_16`).
2. `from_site_adjusted` docstring: says *where* to read SMS/SM1 from, not just
   what they are not.
3. `report()`: `Ss`/`S1` were accepted as constructor metadata and used for
   the §11.6 `S1 >= 0.75` SDC override, but never surfaced in the report —
   they only reached `_parameters()` (spectrum metadata). Now they appear as
   ReportItems labelled "reference only; not scaled", cited to §11.4.3. This
   is display of user input, no derived value, no invented number.

`tests/test_asce7.py` (skill §11, "code rules" category)

- `test_no_site_coefficients_in_7_22` — asserts `ASCE7_22` exposes no `Fa`,
  `Fv`, `Fa_override` or `Fv_override`. Docstring carries the C11.4.3 quote,
  so a future re-add fails with the reason rather than just a red dot. This is
  the direct guard against the change that was requested.
- `test_Ss_S1_are_metadata_only` — supplying Ss/S1 leaves SDS/SD1 unchanged
  and puts Ss in the report.

## Design decisions considered and rejected

- **A raising `ASCE7_22.from_mapped(Ss, S1, ...)` signpost.** Would give a
  better error than `AttributeError`, but naming a constructor after a path
  the standard removed invites the belief that one exists. The module
  docstring is the right place for the redirect. Rejected.
- **Implementing §12.14.8.1's Fa = 1.0/1.4.** Real, but belongs to the
  simplified alternative procedure (a different analysis path with its own
  eligibility limits in §12.14.1.1), not to Chapter 11 spectra. Out of scope
  for this request; would need its own design.

## Quality gate (skill §10) — all clean

Run from the worktree with `PYTHONPATH=<worktree>/src`:

- `python -m pytest -q` -> 507 passed
- `python -m ruff check .` -> All checks passed
- `python -m mypy src/codeSpectra` -> Success, 27 source files
- 3.10 AST parse gate over `src/**/*.py` -> ok

Also spot-checked `report().to_text().encode("cp1252")` on a 7-22 site with
Ss/S1 supplied (skill §5) — the new item text is pure ASCII and encodes fine.

No `tables/*.json` was touched, so the wheel package-data check was not
needed.

## Left undone / offered

The library has no USGS retrieval for any edition (skill §13). I offered to
scope a thin, explicit-opt-in `from_usgs(lat, lon, site_class)` against the
USGS Seismic Design Web Service, since that fetches real values rather than
reconstructing removed ones — but flagged the skill §9 rule that any network
access must be explicit, never implicit.
