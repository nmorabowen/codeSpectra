# The flat Sa = 1.1904 g down to T = 0 is correct — I did not change the code

I checked this against NEC-SE-DS 2015 directly (`NEC_SE_DS_(peligro sismico).pdf`, §3.3.1 and the §10.1 commentary) before touching anything, and `elastic_spectrum()` is behaving exactly as the norm requires. Making the ascending branch the default would have introduced a real unconservatism, so I left the code alone. No files changed.

## Your numbers check out — both of them

For Z = 0.40, soil D, Sierra:

| Quantity | Value |
|---|---|
| Z | 0.40 g |
| Fa, Fd, Fs (Tablas 3/4/5) | 1.20, 1.19, 1.28 |
| eta (Sierra) | 2.48 |
| r | 1.0 |
| T0 = 0.10 Fs Fd / Fa | 0.1269 s |
| Tc = 0.55 Fs Fd / Fa | 0.6981 s |
| TL = 2.4 Fd | 2.856 s |
| Sa plateau = eta Z Fa | **1.1904 g** |
| Z Fa (ramp start at T = 0) | **0.4800 g** |

So 1.1904 g is the plateau and 0.48 g is the foot of the ramp you are reading off Figura 3. Both are right; they are two different curves for two different purposes.

## Why the plateau runs all the way to T = 0

The design spectrum in §3.3.1 is defined over **two** ranges only — the norm says so in the sentence introducing the equations:

```
Sa = eta Z Fa                  for 0 <= T <= Tc
Sa = eta Z Fa (Tc/T)**r        for T > Tc
```

The ascending branch appears *after* those equations, in its own paragraph, and it is fenced in. It applies to dynamic analysis and, in NEC's words, "unicamente para evaluar la respuesta de los modos de vibracion diferentes al modo fundamental" (NEC-SE-DS 2015, §3.3.1). Not the fundamental mode, and not static analysis at all.

The commentary in §10.1 is even more explicit about why. It states that because ductility cannot be relied on to reduce the elastic ordinate at short periods for design, the left ascending branch of the typical elastic response spectrum was **deliberately deleted** for static analysis and for the fundamental mode in dynamic analysis, and the plateau was extended down to periods near zero. That flat top at 1.1904 g is not a missing branch — it is the branch NEC removed on purpose.

Figura 3 does draw the ramp, so your reading of the figure is fair. The figure serves both uses; the text underneath it is what decides which one you may apply.

## If you do want the ramp, it is already there

It is an explicit opt-in rather than a bug:

```python
from codeSpectra.codes.nec import NECSEDS2015

site = NECSEDS2015(zone="V", soil="D", region="sierra")

design = site.elastic_spectrum()                                # plateau from T = 0
modal  = site.elastic_spectrum(include_ascending_branch=True)   # with the T <= T0 ramp
```

```
   T (s)    design    w/ ramp
  0.0000    1.1904     0.4800
  0.0200    1.1904     0.5919
  0.0500    1.1904     0.7598
  0.1000    1.1904     1.0397
  0.1269    1.1904     1.1902   <- T0, the two curves meet
  0.2000    1.1904     1.1904
  1.0000    0.8311     0.8311
```

Note the two curves are identical above T0 = 0.127 s. For the fundamental period of essentially any real building this makes no difference to a single design force — it only shows up in the plot and in very stiff higher modes.

## For the ETABS comparison specifically

Read the ordinate ETABS reports at T = 0 in its response-spectrum function:

- **0.48 g** — ETABS drew the ramped (higher-mode) curve. That is the discrepancy, and codeSpectra is the one following the design provision.
- **1.1904 g** — you already agree, and something else is in play (check that ETABS has Sierra eta = 2.48 and not the Costa 1.80, and that the soil coefficients are the Tabla 3/4/5 values above).

One thing worth flagging before you reconcile them by switching codeSpectra: ETABS applies a single response-spectrum function to **all** modes in a modal RSA. So the ramped curve cannot be used as your ETABS function without also applying the reduced short-period ordinate to the fundamental mode — precisely what §3.3.1 forbids. NEC's split treatment (plateau for the fundamental mode, ramp for the others) is not expressible as one ETABS function.

The defensible move is the plateau curve for everything. It is conservative for the higher modes and exactly correct for the fundamental one. That is why it is the default here.

## Verification

- Confirmed the §3.3.1 restriction and the §10.1 rationale in the NEC PDF, not from memory.
- Full suite green: 505 passed. `ruff` clean. `git diff` empty.
- The existing tests already pin this both ways: `test_plateau_extends_from_zero_by_default`, `test_ascending_branch_starts_at_Z_Fa` (asserts 0.40 x 1.20), and `test_ascending_branch_meets_the_plateau_at_T0`.

If you would rather the library warn when someone plots down to T = 0 without the ramp, or grow a helper that assembles the strictly-correct two-curve modal treatment, say the word — but I would not want either to quietly change what `elastic_spectrum()` returns by default.
