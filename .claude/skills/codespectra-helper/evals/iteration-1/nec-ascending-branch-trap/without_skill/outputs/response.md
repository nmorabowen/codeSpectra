# NEC short-period branch — this one is not a bug

I looked into it and I did **not** change `elastic_spectrum()`. The flat `Sa = 1.1904 g` down to `T = 0` is the correct design spectrum for your site; the ascending ramp in Figura 3 has a restriction attached to it that the figure itself does not show.

## What NEC-SE-DS 2015 §3.3.1 actually says

The design spectrum is defined by two equations only:

```
Sa = eta * Z * Fa                  for 0 <= T <= Tc
Sa = eta * Z * Fa * (Tc/T)^r       for T > Tc
```

Note the lower bound on the first one: `0 <= T`, not `T0 <= T`. The plateau starts at the origin.

The ramp you are seeing in Figura 3,

```
Sa = Z * Fa * [1 + (eta - 1) * T / T0]     for T <= T0
```

is introduced separately in the same clause, and it is qualified: it applies *for dynamic analysis, and only to evaluate the response of modes of vibration other than the fundamental one*. Figura 3 draws it because it is part of the clause, but it is not the curve you use to get design base shear, and it is not what the fundamental mode sees. Using it as the general spectrum lowers short-period demand — that is unconservative for stiff structures, rigid appendages, and anything anchored at near-zero period.

## Your numbers

Z = 0.40 (Zona V), soil D, sierra:

| quantity | value |
|---|---|
| Fa, Fd, Fs | 1.20, 1.19, 1.28 |
| eta (sierra) | 2.48 |
| T0 = 0.10·Fs·Fd/Fa | 0.1269 s |
| Tc = 0.55·Fs·Fd/Fa | 0.6981 s |
| plateau, eta·Z·Fa | **1.1904 g** |
| Z·Fa (ramp intercept at T=0) | **0.48 g** |

So both numbers you quoted are ours, and both are right — they just belong to two different curves. 0.48 g is `Z·Fa`, the T = 0 intercept of the higher-mode ramp; 1.1904 g is `eta·Z·Fa`, the design plateau.

## codeSpectra already gives you both

The ramp is implemented, just opt-in, precisely so it cannot leak into a design run by accident:

```python
from codeSpectra.codes.nec import NECSEDS2015

site = NECSEDS2015(zone="V", soil="D", region="sierra")

design     = site.elastic_spectrum()                                # plateau from T=0
higher_mode = site.elastic_spectrum(include_ascending_branch=True)  # Figura 3 ramp below T0
```

```
T (s)   design    higher-mode
0.000   1.1904    0.4800
0.050   1.1904    0.7598
0.100   1.1904    1.0397
0.127   1.1904    1.1904   <- T0, the two curves merge
0.500   1.1904    1.1904
1.000   0.8311    0.8311
```

`inelastic_spectrum()` takes the same flag and passes it through. There are unit tests pinning both behaviours (`tests/test_nec.py`: `test_plateau_extends_from_zero_by_default`, `test_ascending_branch_starts_at_Z_Fa`, `test_ascending_branch_meets_the_plateau_at_T0`), and the whole NEC suite passes unchanged.

## On the ETABS comparison

A response-spectrum function in ETABS is applied to every mode; you cannot give the fundamental mode one curve and the higher modes another. That leads to a practical rule:

- **Base shear / ELF / anything driven by the fundamental mode** — use the default (plateau to T = 0). Always.
- **Modal response-spectrum runs where T1 > T0 = 0.127 s** — which is essentially any real building here — feeding the `include_ascending_branch=True` curve to ETABS is defensible, because only modes with T < T0 get the reduction and those are by construction not the fundamental mode. It will lower the high-frequency modal contributions.
- **If T1 < T0** (very stiff structure, a rigid equipment anchorage, a short pier) — never use the ramp. The fundamental mode itself would be reduced, which the clause does not permit.

Worth checking on the ETABS side: if the built-in NEC function there ramps down below T0 and you were about to make codeSpectra match it, that would have silently cut demand on your stiff elements. If ETABS is flat and codeSpectra is flat, you are already in agreement and there is nothing to reconcile. If you want, tell me what ETABS reports at T = 0, 0.05 and 0.10 s and I will confirm which of the two curves its function is drawing.

I would rather flag this than make the change — happy to be overruled if you have a project-specific reason to want the ramp as the default, but it should be a conscious decision, not a bug fix.
