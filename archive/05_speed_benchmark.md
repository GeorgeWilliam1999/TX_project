# 05 — Speed / Throughput Benchmark

**Goal:** per-track extrapolation throughput of the current Allen system relative to ours — pure
relative throughput, under heavy scrutiny. **Reference:** Allen's `RungeKuttaExtrapolator.cuh` (the
real per-track adaptive RK, with v8r1 field-map texture lookups), plus `extrapUTT` as the specialised
competitor. **Ours:** the generated NN kernel `PINN_V2_UTT.cuh`.

## Tier-1 — isolated CUDA micro-benchmark (the primary number)
GPU, NVRTC-compiled verbatim Allen device code; 1,000,000 tracks drawn from the real gen-4 (z0,dz,p)
distribution; CUDA-event timing; 200 warm-up iters discarded; 50 timed repeats; fp32; RK with 100 mm
step (mean 82.75 field lookups/track over the population).

| method | µs/track (kernel) | tracks/s | single-warp latency |
|---|---|---|---|
| extrapUTT (polynomial) | 0.00234 | 4.27e8 | 23.6 µs |
| RK + field (incumbent) | 0.00571 | 1.75e8 | 11.3 µs |
| **PINN_v2 (ours)** | **0.00705** | **1.42e8** | 176 µs |

**Headline ratios (kernel-only):** RK ÷ NN = 0.81 (i.e. **the NN is ~1.2× slower than RK**);
extrapUTT ÷ NN = 0.33 (**the polynomial is ~3× faster than the NN**); RK ÷ extrapUTT = 2.44.

## Tier-2 — in-situ Allen (reality check)
Ran against the built Allen (CPU target — no GPU Allen build was available). extrapUTT 0.499 µs/track
vs PINN 22.2 µs/track → on CPU the NN is ~44× slower than the tiny polynomial. **Critically, the
deployed `use_nn_utt` path is a HYBRID:** extrapUTT always runs (for the Jacobian F/Q) and the NN is
*added* for the state — so the NN path is **strictly slower than just running the polynomial.**

## Cross-check caveat (honest)
The plan required the Tier-1 and Tier-2 extrapUTT:NN ratios to agree within 2×. They don't (0.33 vs
0.022) — because **Tier-2 ran on CPU** (the NN matmul amortises on GPU but not CPU; no GPU Allen build
was present). So the absolute GPU comparison rests on Tier-1 alone, without an independent in-situ GPU
confirmation. **Both tiers nonetheless agree on direction:** the NN is slower than both the RK and the
polynomial. A GPU in-situ run remains the one open confirmation.

## Memory footprint (co-metric)
field map texture ~11.5 MB · extrapUTT tables ~0.96 MB · NN ~40 kB · chart ~63 kB. The NN/chart win
decisively on footprint — but footprint alone does not justify a slower, less accurate surrogate.

## Verdict
**The NN speed value proposition is also closed.** It is slower than the very thing it was meant to
replace (RK) and 3× slower than the polynomial, and the deployed form is strictly additive overhead.

## Artifacts
`allen_bridge/bench/` (bench_kernels.cu, bench_rk.cuh, bench_extraputt.cuh, insitu_parkalman.cpp,
assemble_throughput.py, explore_throughput.ipynb); results in `gates/baseline/throughput.json` and
`allen_bridge/bench/results/tier{1,2}_*.json`.
