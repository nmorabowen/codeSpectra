# Iteration 1 — results

Run 2026-08-03 against skill commit `52cc4d5`, library at `bd20bec`.
Three cases, each run twice: once with `SKILL.md` loaded, once with
`.claude/skills/` explicitly withheld. One run per configuration.

## Headline

| | Pass rate | Tokens | Time |
|---|---|---|---|
| with skill | 20/20 (100%) | 124,604 | 515 s |
| baseline | 20/20 (100%) | 107,277 | 445 s |
| **delta** | **+0.00** | +17,327 (+16%) | +70 s (+16%) |

**Every assertion passed in both configurations.** The skill cost ~16% more
tokens and wall-clock and moved no measured outcome. Do not read this as
"the skill works" — read it as "this eval set measures nothing".

## Why the eval set failed to discriminate

**The two trap cases were unwinnable by construction.** Both traps —
ASCE 7-22 having no `Fa`/`Fv` tables, and the NEC ascending branch being
restricted to non-fundamental modes — are already written into the module
docstrings of `asce7_22.py` and `nec_se_ds_2015.py`. The baseline agent is
editing those very files, so it reads the provision on the way past. The eval
was testing whether an agent can read a docstring placed directly in its path.

That is a good property of the codebase and a fatal one for the eval. It is
also a real finding: **for questions the source already documents, the skill
is redundant.**

**The extraction case did not separate either.** Both configurations
independently cross-checked two `pdftotext` render modes, caught the
merged-cell row-grouping trap in NCh433 Tabla 5.1, and stored the missing
`Ro` (unclassifiable row, footnote 3) as `None` rather than fabricating it.
The validation discipline the skill teaches, the model already had.

## The one qualitative difference

On `asce722-fa-fv-trap`, only the with-skill run *verified* its claim rather
than asserting it: it followed the skill's PDF recipe, grepped the 7-22 text
for `Table 11.4` (0 hits anywhere, body or commentary), and surfaced
**Commentary C11.4.3**, which states that `SMS`/`SM1` incorporate site
effects, "eliminating the need for the tables of site factors Fa and Fv of
ASCE 7-16."

The baseline reached the identical verdict from the docstring plus prior
knowledge. Same answer, different epistemic standing. No assertion could see
the difference.

## What the exercise actually produced

Two real defects in shipped library code, both found by **baseline** runs,
neither asked about by any assertion:

- `ASCE7_22(Ss=..., S1=...)` constructed successfully and reported
  `SDS = SD1 = 0.0` — a silently-zero design acceleration that could reach a
  model. Fixed in `ea7f120`.
- `_SDS_input` / `_SD1_input`: declared, never read. Removed in `ea7f120`.

Also surfaced: the pytest config does not run doctests, so a wrong expected
value in a docstring went unnoticed until an agent ran them manually; and
whole-repo `mypy` reports 10 pre-existing errors in test files, since the
gate only checks `src/`.

As a code review this was worth running. As a skill measurement it was not.

## Process faults, recorded so they are not repeated

- **`isolation: worktree` seeds from the repo HEAD, not the working branch.**
  All six agents started on `main` (`c00751e`), which predates the library
  rewrite. Four noticed and self-corrected; the `extract-nch433-R-table`
  baseline did not, and built its answer against the legacy
  `designSpectra.py`. It was re-run pinned to `bd20bec`; the invalid first
  attempt was archived rather than deleted.
- **The editable install resolves `codeSpectra` to one specific worktree**, so
  a bare `pytest` from a sibling worktree silently tests the wrong checkout.
  One agent got a passing run against code it had not written before noticing.
  Pin `PYTHONPATH=<worktree>/src`.
- **The first grading pass produced two false negatives** from brittle keyword
  matching: it missed "is correct — I did not change the code" (no literal
  "not a bug") and validation prose that lived in `notes.md` rather than
  `response.md`. Regraded against response + notes + patch combined. Grade
  against every artefact, not just the headline file.

## What to change for iteration 2

Test what the source does *not* already say:

1. **A standard the repo has never touched** — no docstring to lean on, so
   the extraction playbook has to carry the work.
2. **A code whose spectrum shape does not match `two_period_spectrum`** —
   tests whether the architecture guidance is followed or reinvented.
3. **A live licensing question** — "ship the reader, not the data" appears
   nowhere in the source, only in the skill. This is the largest gap between
   what the skill knows and what the code shows.

Also worth changing: run each configuration more than once. With n=1 per cell
there is no variance estimate, so a tie is weaker evidence than it looks.
