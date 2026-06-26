# Bend-parametrisation study: multi-step unroll & 2nd-order curvature (UT→T)

**What this folder tests.** Whether a *field-free* surrogate can break the ~3 mm UT→T accuracy
floor by changing the bend parametrisation — first by applying the deployable kick head over N
sub-steps (the **multi-step unroll**; Allen-faithful = looping the kernel N times), then by
enriching each step's basis with a **2nd-order curvature term** (~κ²·dz³). Verdict below:
the floor breaks to ~1.25 mm but the field-free route then **plateaus and is exhausted**, and
at the optimised kernel rate the unroll is **not speed-competitive** with extrapUTT.

## Headline — UT→T-subset median (the metric that matters)
`utt_median_dx_um` / `utt_p95_dx_um` from each run's `history.json` `test_final` (full test split):

| approach | steps | epochs (best) | **UT→T median** | UT→T p95 |
|---|---|---|---|---|
| single-shot baseline (h96) | 1 | 120 | 3196 µm | 35.9 mm |
| best single-shot (depth-3 tanh) | 1 | 120 | 3027 µm | 32.9 mm |
| multi-step unroll ×8 | 8 | 250 (214) | 1294 µm | 16.7 mm |
| **multi-step unroll ×16** | 16 | 250 (243) | **1256 µm** | 16.1 mm |
| multi-step unroll ×32 | 32 | 200 (199) | 1258 µm | 16.1 mm |
| unroll ×8 + 2nd-order curvature | 8 | 250 (235) | 1325 µm | 15.8 mm |
| unroll ×16 + 2nd-order curvature | 16 | 250 (236) | 1258 µm | 15.8 mm |

- **The unroll breaks the floor but plateaus at ~1.25 mm** (8→16 buys ~40 µm; 16→32 nothing).
- **The 2nd-order curvature term gives no median gain**: at matched 16 steps, 1258 µm (with) vs
  1256 µm (without) — identical, with only a slight low-p/tail nudge.

## All-pairs by-q/p (200 k test subsample, fp32 — `eval_e1.py`)
| approach | steps | all-pairs median | all-pairs p95 | by-p quartile µm (Q1 low-p → Q4 high-p) |
|---|---|---|---|---|
| single-shot baseline | 1 | 1066 µm | 81.4 mm | 7477 / 2885 / 817 / 210 |
| unroll ×16 | 16 | 276 µm | 30.6 mm | 2247 / 722 / 195 / 49 |
| unroll ×32 | 32 | 256 µm | 27.8 mm | 2007 / 678 / 184 / 45 |
| unroll ×16 + curvature | 16 | 260 µm | 27.5 mm | 1817 / 674 / 187 / 46 |

The all-pairs / low-p view keeps improving with steps (the curvature term helps the low-p Q1
slightly: 1817 vs 2247 µm), but the **hard UT→T long-haul subset is saturated**.

## Speed — at the maximally-optimised kernel rate (V100, `pinn_opt_work`, reproduced 2026-06-27)
Single-step optimised kernels (cluster 4862192, V100-PCIE-32GB, 1 M tracks):
**`pinn_fused` 4.85 ns (bit-exact, beats RK 5.71) · `pinn_h64_fu` 0.91 ns** · extrapUTT 2.34 ns.
The unroll loops the kernel N times, so its optimised cost = **N × single-step**:

| approach | optimised ns/track (h64_fu 0.91 · h96_fused 4.85) | vs extrapUTT 2.34 |
|---|---|---|
| extrapUTT (incumbent) | 2.34 | 1× |
| unroll ×8 | 7.3 (h64) · 38.8 (h96) | 3.1× · 17× slower |
| unroll ×16 | 14.5 (h64) · 77.7 (h96) | 6.2× · 33× slower |

⇒ the multi-step unroll is **slower AND less accurate than extrapUTT** — not deployment-competitive.

## Conclusion
- **Floor broken, route exhausted.** The field-free multi-step unroll takes UT→T from ~3.2 mm to
  ~1.25 mm (2.5×) — confirming the floor was the single-kick parametrisation, not capacity. But
  the lever **plateaus** (16≈32 steps) and the 2nd-order curvature term adds nothing on the
  median. The field-free (Tier-0) parametrisation is therefore **exhausted** — the limit is
  **field information**, not the kick basis or step count.
- **And not competitive.** Even at the optimised kernel rate, the unroll is N× the single-step
  cost → slower than extrapUTT while still ~84× less accurate (extrapUTT 15 µm / 748 µm low-p).
- **Next gain must be field-aware:** field-sample input features (works on the general corpus) or
  a residual on top of extrapUTT (plane-specific). Low-p tail (Q1 ~1.8–2.0 mm, p95 ~16 mm) remains
  the hard part.

## Provenance
- **Models** `…/gen_3/trained_models/`: `e1_unroll{2,4,8}` (Condor 4861428); `e1_unroll{8_long,16,32}`
  (Condor 4861438); `e2_order2_unroll{8,16}` (Condor 4862005). All `pinn_v2` [96,96] tanh
  `kick_scaled_head`, fp32, gen-4 corpus. (`e1_*` = multi-step unroll; `e2_order2_*` = unroll +
  2nd-order curvature — original dir names kept for provenance.)
- **Code** committed `track-extrapolation-pinn @ 0684027` (branch `wave2-retraining`):
  `PINN_v2.n_unroll` (unroll) + `kick_order` (curvature) in `models/architectures.py`; `train.py`
  plumbing; `configs/phaseA/`.
- **Speed** `pinn_opt_work/` (`microbench_opt_v3.py` + `pinn_opt_kernels_v3.cu`), V100 cluster 4862192
  → `results/tier1_opt_v3.json`. Single-step kernels; unroll = N× (see table).
- **Data** gen-4 `train_wave2_deploy.npz` (5,196,722), shared seed-42 test split; 200 k subsample
  (rng seed 7) for the all-pairs curve.
- **Eval** `eval_e1.py` → `results/e1_eval.json`, `figures/fig_e1_vs_qop.png`.
