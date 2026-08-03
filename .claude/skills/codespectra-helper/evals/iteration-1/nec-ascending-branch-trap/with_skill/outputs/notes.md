# Notes — NEC ascending branch report

## Outcome

**No code change.** The reported "bug" is correct, intentional behaviour. `changes.patch` is empty.

## What I did

1. **Loaded the `codespectra-helper` skill.** Its §6 flags this exact situation by name
   ("NEC-SE-DS: the ascending branch is off by default") and states that turning the ramp
   on by default under-estimates short-period demand. That gave me the hypothesis, but I
   treated it as a hypothesis, not proof — the user's claim was specific and plausible.

2. **Found the code.** The worktree I was handed (`agent-a4760b20414ff5ce3`) was checked out
   at `c00751e`, an ancestor commit holding only the old single-file `codeSpectra/designSpectra.py`
   (NCh433 only). No NEC module, no `elastic_spectrum()` at all. The actual library lives at
   `52cc4d5` on `claude/asce-nec-spectrum-library-5739fa`, which is also where the skill's own
   base directory pointed and what the session's git status claimed to be the current branch —
   a harness checkout mismatch. Since `c00751e` is a strict ancestor of `52cc4d5`, I fast-forwarded
   my worktree branch (`git merge --ff-only 52cc4d5`). Non-destructive, no commit created,
   working tree clean afterwards. Without this there was no `elastic_spectrum()` to discuss.

3. **Reproduced the numbers.** `src/codeSpectra/codes/nec/nec_se_ds_2015.py` with
   Z=0.40 / soil D / Sierra gives Fa=1.20, Fd=1.19, Fs=1.28, eta=2.48, r=1.0,
   T0=0.1269 s, Tc=0.6981 s, TL=2.856 s, plateau = 2.48 x 0.40 x 1.20 = **1.1904 g**.
   With `include_ascending_branch=True`, Sa(0) = Z Fa = **0.4800 g**.
   Both of the user's figures reproduce exactly, from the two different curves.

4. **Verified against the norm rather than the skill.** Per the skill's prime directive
   (never assert a code value you have not read), I extracted the NEC PDF at
   `C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos\Ecuador\NEC\NEC_SE_DS_(peligro sismico).pdf`
   with `pdftotext -layout` and read §3.3.1 in place. Two decisive findings:

   - §3.3.1 introduces the design spectrum as valid over **two** ranges (0..Tc, >Tc). The
     ascending branch is a separate later paragraph restricted to dynamic analysis and,
     verbatim, "unicamente para evaluar la respuesta de los modos de vibracion diferentes
     al modo fundamental."
   - The §10.1 commentary is stronger still: it says the left ascending branch of the typical
     elastic spectrum was **eliminated** for static analysis and for the fundamental mode in
     dynamic analysis — because ductility cannot reduce the short-period elastic ordinate for
     design — and the plateau deliberately extended to periods near zero.

   So the flat top is not a missing branch; it is a branch NEC removed on purpose. Changing
   the default as requested would have made every short-period design spectrum unconservative
   by up to a factor of eta (2.48 here).

5. **Confirmed the repo is already clean and already tested.** `pytest` 505 passed,
   `ruff check .` clean, `git diff` empty. `tests/test_nec.py` already pins all three facts:
   `test_plateau_extends_from_zero_by_default`, `test_ascending_branch_starts_at_Z_Fa`
   (0.40 x 1.20), `test_ascending_branch_meets_the_plateau_at_T0`.

## Why I added nothing

The behaviour is documented in three places already (module docstring, `elastic_spectrum`
parameter doc, README) and pinned by tests. Adding a warning or a doc paragraph would be
gold-plating a correct implementation. I offered both as options in the reply instead of
doing them unasked.

## Engineering point worth keeping

ETABS applies one response-spectrum function to all modes in a modal RSA, so the ramped curve
cannot be used there without also applying the reduced short-period ordinate to the fundamental
mode — exactly what §3.3.1 forbids. NEC's split treatment is not expressible as a single ETABS
function, which makes the plateau curve the right thing to feed ETABS: exact for the fundamental
mode, conservative for the rest. This is the practical reason the default matters, and it is the
most likely explanation for what the user is seeing on the ETABS side.

## State left behind

Worktree fast-forwarded to `52cc4d5`, working tree clean, nothing committed, nothing modified.
