# Notes — ASCE 7-22 Fa/Fv request

## Workspace surprise (resolved first)

The worktree I was handed (`agent-a3d575e97c19b3681`) was checked out at `c00751e`,
the *pre-rewrite* state of the repo: a single `codeSpectra/designSpectra.py` holding
only an `NCh433` class. No ASCE code at all. The library the user is describing lives
on branch `claude/asce-nec-spectrum-library-5739fa` (`52cc4d5`), checked out in a
sibling worktree.

I ran `git reset --hard claude/asce-nec-spectrum-library-5739fa` in my own worktree to
sync to the branch under discussion. Nothing was lost — `worktree-agent-a3d575e97c19b3681`
was a scratch branch pointing at a commit reachable from `main`. No commits made.

Per the run constraint I did not read `.claude/skills/`. One line of `SKILL.md` surfaced
incidentally in a repo-wide grep hit; I did not open the file, and the conclusion below
was reached independently from the source and from the standard.

## The request rests on a false premise

The user asked me to "add the Fa/Fv site coefficient tables to asce7_22.py". Those
tables do not exist in ASCE 7-22. Deleting the site coefficients was one of the
headline changes of the 7-22 seismic ground motion provisions: Tables 11.4-1 and
11.4-2 are gone, and §11.4.3 has `SMS`/`SM1` read directly from the USGS Seismic
Design Geodatabase for the applicable site class, with site amplification already
embedded in the multi-period spectra.

Writing tables into `asce7_22.py` would have meant either relabelling 7-16's values
with a 7-22 citation, or inventing numbers. Both are exactly the failure this library
is architected against — every value in it carries a `ClauseRef`, so a fabricated
table would propagate a false citation into user reports.

`asce7_22.py`'s module docstring already stated this plainly. The code was right;
the request was not.

Two additional reasons the borrow is incoherent, not merely unofficial:
- 7-22 added Site Classes BC, CD, DE and made the default class "most critical of
  C/CD/D". Three of eight classes have no 7-16 coefficient; the default has no analogue.
- 7-22 `Ss`/`S1` come from new GMMs on a multi-period basis, so 7-16 `Fa` × 7-22 `Ss`
  mixes two hazard characterizations even for Site Class D.

## The real defect underneath the complaint

The user's frustration was pointing at something genuine one layer down. `Ss`/`S1` are
optional metadata fields on the `ASCE7_22` dataclass, so the 7-16-shaped call bound
them happily and left `SMS`/`SM1` at their `0.0` defaults:

    ASCE7_22(Ss=1.5, S1=0.6, site_class='D')  ->  constructs fine, SDS = SD1 = 0.0

`.SDS` returned `0.0` with no warning. The only complaint arrived much later and from
the wrong place — `InvalidInput: A tabulated spectrum needs at least two points` — which
says nothing about the real problem. A silently-zero `SDS` in a seismic design library is
a serious footgun; it can survive into a model.

**Fix:** a guard in `__post_init__` rejecting construction when the active basis has no
usable input (MULTI_PERIOD with <2 ordinates, or TWO_PERIOD with `SMS <= 0`). The message
is tailored when `Ss`/`S1` were supplied — it names the deleted tables, cites §11.4.3,
names both supported constructors, and describes the explicit 7-16 handoff. Only direct
construction is affected; both classmethods already validated their inputs.

## Changes

- `src/codeSpectra/codes/asce7/asce7_22.py`
  - `NO_SITE_COEFFICIENTS` constant + `_missing_site_adjusted_message()` helper.
  - `__post_init__` guard against the incomplete/7-16-shaped call.
  - Module + class docstrings: `Ss`/`S1` documented as reporting metadata only.
  - `report()` now emits `Ss`/`S1` items when supplied, labelled "reported only;
    no Fa applies" — visible provenance without implying a conversion.
- `tests/test_asce7.py` — 5 tests: rejection of the 7-16-shaped call, rejection of
  bare construction, metadata retention + §11.6 `S1 >= 0.75` override, and the
  documented 7-16 handoff.
- `README.md` — metadata-only semantics, and the explicit 7-16 borrow with a clear
  non-compliance warning.

## Judgment call: no `from_ss_s1` constructor

I considered a clearly-named `ASCE7_22.from_asce7_16_coefficients(...)`. I rejected it.
A blessed constructor reads as a 7-22 provision, which is the precise confusion that
prompted the request, and it would silently fail to serve BC/CD/DE. The two-line
`ASCE7_16(...) -> .SMS/.SM1 -> from_site_adjusted(...)` handoff is equally short, is
self-documenting at the call site, and is now README'd and tested. The right long-term
ergonomic fix is a USGS geodatabase client (`from_usgs(...)`), which I flagged.

## Verification

- Full suite: 509 passed.
- Doctests on `asce7_22.py` via `pytest --doctest-modules`: pass.
- Error message and report output inspected by hand.

## Environment gotcha

`codeSpectra` is pip-installed editable pointing at the `asce-nec-spectrum-library-5739fa`
worktree, so a bare `pytest` from any other worktree tests *that* checkout, not the edited
one. My first run showed 3 failures from stale code with tracebacks pointing at the sibling
path. Ran everything under `PYTHONPATH=<this worktree>/src` thereafter. Flagged to the user;
I did not alter their global install.

## Not done

- No commits, per instructions.
- `_SDS_input` / `_SD1_input` on `ASCE7_22` are dead fields (declared, never read).
  Adjacent to my edit but a separate concern; flagged as a follow-up task rather than
  bundled in.
