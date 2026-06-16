# 03 — The κ & Field Discovery (the central finding)

## What was wrong
Every corpus from gen-1 onward integrated the Lorentz equations with a bending constant **1000× too
weak**. The code used `κ = 1e-6 · qop` where the physics requires `κ = 1e-3 · qop`.

### The unit derivation
```
dθ/ds = 0.299792458 [GeV/(T·m)] · B[T] / p[GeV]                 per metre
      = 2.99792458e-4 · B · (q/p)[1/GeV]                        per mm
      = 0.299792458   · B · (q/p)[1/MeV]                        per mm
with Allen qop = 299.792458 · (q/p)[1/MeV]:
dθ/ds = 1.0e-3 · qop · B                                        per mm   ← correct
```
The legacy code paired the constant `2.998e-4` (which belongs with q/p in **1/GeV**) with q/p in
**1/MeV** → off by 1000. The original gen-3 comment even *derived* `1e-6` by matching the legacy
constant, so the bug reproduced itself across generations through "consistency with the past."

## Why no internal check caught it
Training data, evaluation truth, the A4 Jacobian reference, and the PINN PDE residual **all** used the
same κ. Every split agreed; every gate passed. The independent analytic-flattening line even
*empirically calibrated* κ₀ = 1.0117e-6 with R²=0.9992 — a perfect fit to the wrong world. The only
instrument that could object was one **not built from the corpus**: the production extrapUTT polynomial.
On first contact its kick disagreed by three orders of magnitude.

Empirically, at the low-momentum quartile the corpus slope kick was **0.35 mrad** where physics demands
**0.59 rad**.

## The field and polarity, also wrong
- The corpus used the toy `twodip.rtf` map. The production coefficients were fit to **LHCb FieldMap
  v8r1**. On identical PV-pointing inputs the two fields' UT→T endpoints differ by **median 473 mm**.
- The legacy loader's "polarity −1" returned **+By**; real MagDown is **−By**. Tracks bent the wrong
  way as well as too weakly.

## The fix and the external calibration
- `κ = 1e-3` (in `models/architectures.py` and `core/rk4_propagator.py`, with the full derivation in
  comments).
- Canonical field = **v8r1 down**, raw sign (MagDown By<0), loaded from the CVMFS
  `field.v8r1.down.bin` Allen consumes (`core/field_v8r1.py`).
- Polarity: extrapUTT pairs with **m_polarity = −1**.

Result on PV-pointing tracks at the UT→T plane (z 2665→7826):

| configuration | median |Δx| | median |Δtx| |
|---|---|---|
| old corpus vs polynomial | ~370 mm | ~0.17 rad |
| κ fixed, twodip field | 13.6 mm | 5.6 mrad |
| **κ fixed, v8r1 field, pol −1** | **15 µm** | **5.7 µrad** |

The whole stack — integrator, field, units, conventions — now reproduces the production polynomial to
its own fit-residual level. The extrapUTT bake-off harness became the project's **permanent external
truth check**: no corpus trains a model before passing it.

## Lessons (the publishable methods content)
1. **Self-consistency is not correctness.** Two independent research lines, 13 reports, dozens of
   gates — all internally perfect, all physically void.
2. **One external baseline beats any number of internal ones.** The incumbent-baseline measurement was
   prioritised precisely because it was the only check not derived from the corpus; it fired on first
   use.
3. **Constants deserve derivations, not citations of prior code.** The fix carries the unit derivation
   in the source.
