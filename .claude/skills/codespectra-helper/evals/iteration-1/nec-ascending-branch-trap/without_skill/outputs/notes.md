# Notes — NEC ascending-branch report

## Outcome

No code change. The reported "bug" is the documented, intended behaviour; changing it would have
introduced an unconservative error. `changes.patch` is empty.

## Worktree situation (worth flagging)

The worktree I was handed (`.claude/worktrees/agent-a561733a50b7cb5c8`) was checked out on its own
branch `worktree-agent-a561733a50b7cb5c8`, sitting at `c00751e` (= `main`). At that commit the repo
contains only the old single-file `codeSpectra/designSpectra.py` (NCh433 only) — no NEC code, no
`elastic_spectrum()`. The actual library lives on `claude/asce-nec-spectrum-library-5739fa`
(`52cc4d5`), which is checked out in a different worktree and so could not be checked out here.

I ran `git reset --hard bd20bec` on my own throwaway worktree branch to materialise the library.
`bd20bec` is the parent of `52cc4d5`; the only thing `52cc4d5` adds is `.claude/skills/...`, which I
was told to treat as non-existent, so resetting one commit short gave me identical library content
without materialising that directory. Nothing was committed, and `git diff` is clean.

## Investigation

1. Read `src/codeSpectra/codes/nec/nec_se_ds_2015.py`. The module docstring and the
   `elastic_spectrum()` docstring already state the rule explicitly: the plateau `Sa = eta Z Fa`
   runs over `0 <= T <= Tc`, and the ramp `Sa = Z Fa [1 + (eta-1) T/T0]` is restricted by
   NEC-SE-DS 2015 §3.3.1 to dynamic analysis, modes other than the fundamental. It is exposed as
   the opt-in keyword `include_ascending_branch` (default `False`), threaded through
   `inelastic_spectrum()` as well.
2. Reproduced the user's site (Z=0.40 / Zona V, soil D, sierra):
   - Fa=1.20, Fd=1.19, Fs=1.28, eta=2.48
   - T0=0.1269 s, Tc=0.6981 s, TL=2.856 s
   - plateau = eta*Z*Fa = 1.1904 g (the value the user saw)
   - Z*Fa = 0.48 g (the "0.48 g at T=0" the user read off Figura 3)
   Both of the user's numbers are produced by the library; they belong to two different curves.
3. Ran `pytest tests/test_nec.py` — 156 passed. Existing tests already pin both behaviours:
   `test_plateau_extends_from_zero_by_default`, `test_ascending_branch_is_off_by_default`,
   `test_ascending_branch_starts_at_Z_Fa`, `test_ascending_branch_meets_the_plateau_at_T0`.

## Why not "fix" it

Figura 3 draws the ramp because the ramp is part of §3.3.1, but the figure carries none of the
clause's qualifier. Making the ramp the default would reduce `Sa` for every period below
T0 = 0.127 s — unconservative for stiff structures, rigid appendages, and zero-period anchorage,
and it would also reduce the fundamental mode for any structure with T1 < T0, which the standard
does not allow. The existing design/higher-mode split is correct.

## What the reply adds beyond "not a bug"

- The tabulated comparison of the two curves so the user can see both of their numbers explained.
- The practical ETABS rule: a response-spectrum function is applied to all modes at once, so the
  ramp curve is only defensible in ETABS when T1 > T0 (then only non-fundamental modes are
  reduced); never for base shear/ELF, never when T1 < T0.
- An offer to identify which curve the ETABS built-in NEC function is drawing, from its ordinates
  at T = 0, 0.05, 0.10 s.
