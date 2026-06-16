# ADR 0011 — Admissibility of the analytic-flattening chart under the ADR 0009 replacement criterion

* **Date:** 2026-06-11
* **Status:** **Proposed** (ruling required — G. Scriven)
* **References:** [0009](../../../gen_3/For_Allen/docs/decisions/0009-replacement-goal-restated.md)
  (replacement goal restated), `experiments/flattening/PLAN.md`, the F1/F2/F3 write-ups
  in the "Analytic Flattening" Notion project, `Allen/device/kalman/ParKalman/include/ParKalmanMethods.cuh:287`
  (`extrapUTT`, the production incumbent).

## Context

The analytic-flattening chart (`experiments/flattening/charts/chart.py`) reaches
**12.1 µm median / 267 µm p95 / 884 µm p99** on the UT→T pool with **zero trained
parameters** — 25× better than the locked neural candidate `pinn_v2_small_v1`
(293 µm). At inference it evaluates:

1. a fixed `N_s = 80`-node quadrature of the dipole kick along the straight chord, and
2. an even-multipole reconstruction `B̂_y(x,y,z) = Σ_{a,b even} c_ab(z) (x/X_N)^a (y/Y_N)^b`
   read from **frozen 1-D tables** `c_ab(z)` (28 even terms × a 25 mm z-grid, **63.5 kB**).

It performs **no 3-D field-map lookup** (the 957k-point map is never touched at
inference) and **no adaptive Runge–Kutta loop**.

ADR 0009 defined the deployment candidate as a *full neural replacement* —
"with no field-map lookup and no analytic Lorentz step." Read literally, the
chart's dipole kick is "an analytic Lorentz step," which would make it
inadmissible. This ADR exists because that literal reading conflicts with the
**stated intent** of ADR 0009 and with the nature of the **incumbent the chart
would replace**.

### What ADR 0009 was actually ruling out

ADR 0009's Context names the prohibited object precisely:

> *A hybrid evaluates the analytic Lorentz right-hand side **and the magnetic
> field map** at inference.*

The intent was to forbid the **expensive** hybrid: the one that reads the full
3-D field map and integrates the Lorentz RHS through an adaptive RK loop at
inference (the very cost the project set out to eliminate). The chart does
neither. It reads ~kB of frozen constants and runs a fixed-cost, branch-free
quadrature — computationally in the same class as a small neural forward pass,
not the RK extrapolator.

### The incumbent is already a frozen-coefficient analytic kick map

The production map the chart replaces, `extrapUTT`
(`ParKalmanMethods.cuh:287`), is **itself** exactly this class of object. Its
`dev_UTT_META` block carries an **order-2 even-multipole bend**:

```
bendx = BENDX + BENDX_X2·(x/z_i)² + BENDX_Y2·(y/z_i)²
bendy = BENDY_XY·(x/z_i)(y/z_i)
```

a momentum-scaled kick `fq = qop·PMIN`, a straight-line base step
`x += tx·(z_f − z_i)`, and **binned polynomial coefficient tables**
(`kalman_params->x00, x10, x01, tx00, …`) evaluated via `compute_state<DEG>()`.
That is structurally identical to the chart — *a frozen-coefficient analytic
kick with an even-multipole bend* — only at **order 2** in (x,y) instead of the
chart's **order 12**, and with a binned-polynomial residual instead of a
z-tabulated one. The production HLT1 Kalman filter already ships this object on
GPU and treats its coefficient tables as admissible constants.

## Decision (proposed)

> **The analytic-flattening chart is admissible as a deployment candidate.** Its
> frozen 1-D tables `c_ab(z)` are *weights by another name* (a fitted
> parameterisation, functionally identical to network weights), and its fixed
> quadrature is the *architecture*. ADR 0009's prohibition is hereby read by its
> intent — **no 3-D field-map lookup and no adaptive-RK loop at inference** —
> which the chart satisfies. Admissibility is by **precedent**: the chart is the
> same class of object as the incumbent `extrapUTT` it replaces, at higher
> fidelity.

Admissibility is conditional on the chart meeting the same engineering bars as
any neural candidate:

1. **A4 Jacobian gate.** The chart must pass the Allen Kalman-Jacobian
   constraint (fp64 reference in `For_Allen/artifacts/phase1a/`). Its analytic
   ∂(out)/∂(state, q/p) is in fact a strength — it is exact and cheap, not
   autograd-approximated.
2. **Budget accounting.** The 63.5 kB of tables count against the deployment
   weight/constant-memory budget exactly as network weights do (cf. the 64 kB
   GPU constant-memory ceiling already tracked for the neural blob).
3. **Single fixed-cost forward pass.** No adaptive iteration, no data-dependent
   branching on the field; the per-call cost must be a constant (2 table reads +
   the fixed quadrature), reported alongside `extrapUTT` and the neural blob.
4. **Same frozen test set + polarity convention** (MagDown, m_polarity = −1) as
   all other candidates.

## Consequences

* If accepted, the chart becomes a first-class deployment candidate alongside
  `pinn_v2_*`, and Phase F5 (blob carries the `c_ab` tables; CUDA header;
  R6 throughput/parity gates) is unblocked.
* The neural residual network (F2) is **not** part of the candidate — F2 showed
  it is not learnable and not required; the admissible object is the
  zero-parameter chart alone.
* A clean numeric **bake-off vs `extrapUTT`** is still owed (Phase F4). The
  existing P0.1 scoring (`gen_3/paper_p0/P0p1_baseline_verdict.json`) is
  **unreliable** — it reports a 369 mm `extrapUTT` median with km-scale tails,
  i.e. the standalone polynomial harness is mis-wired or evaluated out of its
  valid (p, angle) domain. The bake-off must be re-run with a corrected harness
  before the comparison table is published; **this ADR does not depend on that
  number** (it rests on the structural-precedent argument, not on out-scoring
  a broken baseline).

## Reversibility

This ADR adds an admissible candidate; it deletes nothing and changes no Allen
code. If the chart fails the A4 gate or the budget bar, it is simply dropped
from the candidate set and the neural path (ADR 0009) continues unchanged. The
ruling can be revised by a successor ADR without code impact.
