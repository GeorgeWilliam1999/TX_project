# 06 — The Analytic Chart (flattening) Programme

> Live code carried forward in `../Chart/`. This file is the v1 record + honest status.

## The idea
Instead of learning the curved endpoint map, **analytically flatten** the curvature. For an idealised
on-axis field `B = By(z)·ŷ`, the canonical momentum `Px = px + q·F(z)` (with `F(z)=∫By dz`) is conserved
and the field-weighted measure τ(z) ∝ F(z) makes the leading dynamics constant-coefficient. The real
field's transverse dependence breaks integrability and is the only learning target. Inference uses
small frozen 1-D tables `c_ab(z)` (multipole coefficients) + a fixed chord quadrature — **no field map,
no RK loop** (argued admissible under the replacement criterion, ADR 0011: "weights by another name").

## Weak-field results (toy field — proof of principle)
- **F0:** field-integral tables + κ₀ calibration (κ₀=1.0117e-6 — note: this is the *toy*-field value,
  itself a symptom of the κ bug).
- **F1:** the "ladder" — a single 120-sample chord quadrature of the 3-D map (rung-1.5) hit **5.7 µm**
  median on UT→T, beating every NN 25–51×.
- **F3:** the deployment chart — even-multipole `c_ab(z)` tables (O12, anisotropic window, σ_w=3000),
  **12.1 µm median, 0 trained params.**
- **F2:** a residual NN on top of the chart — verdict CLOSED (the residual is a path functional, not
  learnable from the endpoint state).
- **F3.2 (Maxwell expansion), F3.3 (rung-2 path iteration)** — both CLOSED as dead ends/negligible.

These were genuinely strong — **on the smooth toy field.**

## Physical-field result (06-14) — the chart also loses
Rebuilt on the real **v8r1** field (`charts/build_chart_v8r1.py` → `chart_tables_v8r1.npz`,
self-consistent κ₀≈1.0e-3). Bake-off vs the incumbent on PV-pointing tracks:

| candidate (real v8r1 field) | median dx | p95 | p99 |
|---|---|---|---|
| straight line | 237 mm | 1.37 m | 1.77 m |
| chart, O12 rebuilt on v8r1 | **4,725 µm** | 137 mm | 301 mm |
| chart, best-tuned (O20) | 3,868 µm | 114 mm | — |
| **extrapUTT (pol −1)** | **14.9 µm** | 2.2 mm | 11.7 mm |

**extrapUTT beats the chart ~260–320×.** The chart is worse at *every* momentum quartile, including
the highest-p (791 µm where there is almost no bend). The 12.1 µm headline was a **toy-field artifact.**

## Root cause (important for any rescope)
The chart's global even-multipole basis represents the **toy** field's By to ~2 % but the **real**
v8r1 field only to ~8 % over the chord region → an ~8 %-wrong kick → mm-scale endpoint error. The real
field carries **localised transverse structure** (fringe/edge fields; on-axis |By| climbs −1.03 T at
y=0 to −2.02 T at y=1800 non-monotonically; raw peak **4.94 T at (−1600,1100,4300)**) that a global
degree-≤20 polynomial cannot fit (Runge ringing at the window edges). **Raising the order doesn't help**
(O12→O20: 4725→3868 µm). extrapUTT wins because it fits a **degree-9 polynomial per 60×50 (x,y) bin** —
it adapts *locally*.

## Is there a path? (feasibility probe, 06-15)
A **localised** representation (coarse 3-D grid + trilinear) cuts the median field-reconstruction error
~10× vs the global multipole (0.37 % at 52 kB, still ~18× smaller than extrapUTT's ~960 kB tables),
which would likely move the chart median from ~4.7 mm toward ~0.1–1 mm. **But parity with extrapUTT
(15 µm) is NOT demonstrated**, and a localised chart converges toward extrapUTT's own design — i.e. "a
cheaper extrapUTT." Worth pursuing **only** if it can match extrapUTT accuracy at a real footprint win.

## Status for the rescope
The flattening *principle* is sound; the *global-multipole representation* is the weak link. Carry the
code forward (`../Chart/`), but treat a localised-representation redesign as a new, scoped phase with a
clear bar: **match extrapUTT's ~15 µm at ≪960 kB**, or don't deploy.
