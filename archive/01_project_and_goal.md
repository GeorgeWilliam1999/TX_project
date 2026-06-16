# 01 — Project & Goal

## The physics task
A charged track in LHCb is described by a 5-component state at a z-plane:
`(x, y, tx, ty, q/p)` — position, slopes `tx=dx/dz, ty=dy/dz`, and signed inverse momentum.
**Extrapolation** propagates that state from `z0` to `z0+dz` through the dipole field, integrating

```
dtx/dz = κ·N·[ tx·ty·Bx − (1+tx²)·By + ty·Bz ]
dty/dz = κ·N·[ (1+ty²)·Bx − tx·ty·By − tx·Bz ]
N = sqrt(1+tx²+ty²),   κ = (q/p)·c_light
```

## The deployment context (Allen)
- **Allen** is LHCb's GPU HLT1 software. Its Kalman track fit calls an extrapolator per track, per
  step — the dominant repeated cost. The general method is the **RungeKuttaExtrapolator** (adaptive,
  reads the 3-D field map).
- For one specific, hard step — **UT→T** (UT tracker at z≈2665 mm → T stations at z≈7826 mm, straight
  through the magnet) — Allen also has **extrapUTT**: a parametrised polynomial (~19 effective params,
  degree-9 polynomial per 60×50 (x,y) bin) fit to the real field. Fast and accurate.

## v1's goal (as originally scoped)
A **drop-in replacement for the RK extrapolator** that uses **no field map at inference** — a compact
neural network (the "PINN_v2") or an analytic chart — small enough for Allen's constant-memory budget
(~64 kB), accurate enough for the Kalman fit, and faster than RK.

## The "replacement criterion" (ADR 0009)
A true replacement must do **no field-map lookup and no Lorentz-RHS evaluation at inference** —
otherwise it is just RK by another name. This ruled out hybrids (e.g. NeuralRK4) as deployment
candidates. The chart's small frozen tables were argued to be "weights by another name" and thus
admissible (ADR 0011) — see `06_chart_programme.md`.

## Why UT→T became the benchmark
It is the single hardest step (full magnet crossing) **and** the one place a production polynomial
already exists — so it is both the toughest test and the one with a real, external yardstick
(extrapUTT). That yardstick is ultimately what exposed the κ bug.

## What changed by the end of v1
The benchmark against the real incumbent showed the goal as originally framed is largely already
solved for UT→T (extrapUTT is fast + 15 µm accurate), and our surrogates don't beat it on the real
field. The rescope must therefore start from **what Allen actually needs that isn't already solved** —
e.g. the *general* extrapolation steps (arbitrary dz, where no polynomial exists), or a different
value axis (footprint, maintainability) — rather than re-attacking UT→T. That is the next phase.
