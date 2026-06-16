# Extrapolator benchmark harness (Phase 0 foundation)

**Rebuilt from source 2026-06-16.** Every integrator is grounded directly in the
read-only Allen tree with file:line — no reuse of prior/archived scripts.
Companion to the Notion scoping to-do **Allen Scoping — Extrapolator Benchmark
To-Do & Implementation Plan (2026-06)**. Nothing here writes into the READ-ONLY
`Allen` / `TE_stack` trees.

## Modules
| file | role |
|---|---|
| `fieldmap.py` | loads `magfield.bin` (parser: `MagneticField.cpp:54-87`); Allen-faithful trilinear (`MagneticField.cuh:38-89`); `qop = C_LIGHT·q/p[GeV]` ≡ Allen `c_light·q/p[MeV]`. |
| `integrators.py` | **truth** = DOP853 (fp64); **deployed incumbent** = `rk_allen_cashkarp` (fixed-step Cash-Karp with Allen's off-by-one, `ExtrapolateStates.cu:46-49` + `RungeKuttaExtrapolator.cuh:25-45` + `ButcherTableau.cuh:74-107`); `buggy=False` = corrected CK; `rk_nystrom_fast` = the ttrack-chain extrapolator; convergence study; finite-diff Jacobian + `frob_rel`. |
| `selftests.py` | **κ-guard** — fail-loud: qop-convention equivalence, c_light, raw By<0, peak −1.048 T, ∫By·dl=−3.733 T·m, static+dynamic pT kick ≈1.12 GeV, truth convergence. |
| `scorer.py` | paired residuals (candidate−truth) → µm/µrad; momentum stratification + easy/hard quartiles; bootstrap CIs; fp32 floor. |
| `sampler.py` | general-step input distribution (random z0, Δz; log-uniform 1/p; LHCb acceptance). Phase-1 will swap in real MC. |
| `run_phase0.py` | orchestrates → `phase0_report.json`. |

## Run
```
python3 run_phase0.py     # ~10 s: κ-guard, incumbent-vs-truth, bug cost, step sweep
python3 selftests.py      # κ-guard only
```

## What Phase 0 established (rebuilt)
- Truth (DOP853) reproduces every external anchor (κ-guard green) and is
  cross-confirmed by RK45/Radau/hand-RK4 → trustworthy ground truth.
- **The deployed `extrapolate_states_t` RK is degraded by an off-by-one stage
  loop** (`RungeKuttaExtrapolator.cuh:32`, `for i<stage-1`). Faithfully
  reproduced: at the production `dz=100 mm` it gives **~250 µm median / ~1.7 mm
  p95** vs truth, worst at low p (≈1.3 mm at 3–5 GeV). A *corrected* Cash-Karp at
  the same cost is **~0.10 µm** — the bug inflates the error ~2485× and drops the
  method to ~1st order (error ∝ dz across the step-size sweep).
  **Implication:** the cheapest general-step win is likely *fixing the bug*
  (same accuracy at ~¼ the field evals, or 2485× accuracy at equal cost) — to be
  confirmed in the writable clone `/data/bfys/gscriven/Allen_rw`.
- fp32 noise floor ≈ 0.08 µm — far below any target.

> Sweep timings are pure-Python reference, **not** GPU throughput. Real
> throughput is Tier-1 CUDA + Tier-2 in-situ HLT1 (Phase 6), in `Allen_rw`.

## Not yet wired
- extrapUTT external baseline (UT→T-specific; deferred — focus is general-step).
- Vectorised truth-grade bulk data-gen (`at_batch`) for million-track datasets.
- Real-MC track extractor (Phase 1).
