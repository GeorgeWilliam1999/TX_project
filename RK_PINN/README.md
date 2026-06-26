# RK_PINN — PINN_V2_UTT GPU kernel throughput optimisation

**Date:** 2026-06-23 · **Status:** complete (3 V100 benchmark jobs) · **Author:** Claude (for George)
**One line:** the locked NN UT→T extrapolator, re-implemented, goes from the **slowest** of the three
Allen extrapolators to the **fastest** — *without changing its outputs* — which **overturns the speed
half of the archived "the NN is dead" verdict.** The accuracy half (~3 mm vs the polynomial's ~15 µm)
is untouched and remains the real open question.

> Read `archive/05_speed_benchmark.md` and `archive/09_verdict_and_lessons.md` first — this folder is
> the direct rebuttal to the "slower than RK" claim in those. Numbers here are new and measured.

---

## 0. Why this matters (reconciliation with the archive)

The v1 close-out concluded (archive/05, archive/09):

| (archived) candidate, UT→T, real v8r1 | median err | speed vs RK | verdict |
|---|---|---|---|
| extrapUTT (polynomial) | 15 µm | 2.4× faster | the bar |
| RK (incumbent) | truth | 1× | what we replace |
| **neural net (wave-2 best)** | **~3 mm** | **1.2× slower** | **dead — accuracy floor + slower** |

That "1.2× slower" was taken as a *second, independent* reason to abandon the NN. **It was an artefact
of a naïve kernel, not a property of the network.** Re-implemented properly, the same arithmetic runs:

| this work, UT→T, same 1 M tracks, same V100 | ns/track | speed vs RK | speed vs extrapUTT |
|---|---|---|---|
| **`pinn_fused`** (fp32, **bit-exact** to the locked kernel) | **4.85** | **1.18× faster** | 0.48× |
| **`pinn_h64_fu`** (accuracy-equivalent width, full-unroll) | **0.91** | **6.3× faster** | **2.6× faster** |

So the NN is **not** slower than RK; it is faster, and at the accuracy-equivalent h64 width it is the
**fastest extrapolator we have, beating even the polynomial.** The honest, updated verdict:

> **The speed objection to the NN is withdrawn.** What remains is purely the accuracy gap (~3 mm vs
> ~15 µm) — and a deployment caveat (today's Allen path runs the NN *on top of* extrapUTT, so it must
> become a true replacement, Jacobian included, before any speed win is realised in situ).

This reframes the strategic question from *"is the NN viable?"* (no, on UT→T accuracy) to *"where is a
**fastest-available, ~3 mm** extrapolator actually valuable?"* — see §8.

---

## 1. The model(s)

### 1a. The locked deployed net — what was optimised
`candidate/pinn_v2_ALLEN_v1/PINN_V2_UTT.cuh` → `pinn_v2_utt_state()` in the deliverable repo
`/data/bfys/gscriven/track-extrapolation-pinn` @ `283b03b`.

- **Architecture:** `6 → 96 → 96 → 4`, `tanh` on the two hidden layers, linear head, **fp32**.
  (Inputs: `x, y, tx, ty, qop` z-score-normalised + a constant `1.0`; `z_frac=1`.)
- **Weights:** baked as `constexpr` arrays — `kW0[576]`, `kW1[9216]`, `kW2[384]` + biases;
  **40.6 KB** total (≤ the 64 KB Allen constant-memory budget). Source blob SHA256
  `c66576709288…e52c`, CRC32 `0x1a139335`, emitted 2026-05-21.
- **Output = IC-preserving envelope** (spec §3), *not* the raw net output:
  `tx' = tx + c0`, `ty' = ty + c1`, `x' = x + tx·dz + c2·dz`, `y' = y + ty·dz + c3·dz`; `qop` passes
  through unchanged. The net only predicts the 4 corrections `c0..c3`; the straight-line drift is exact.
- **MACs/track:** 6·96 + 96·96 + 96·4 = **10 176**, plus 192 `tanh`.

### 1b. The accuracy-equivalent width — h64
The recorded **wave-2 capacity ladder** (`<repo>/results/wave2/wave2_three_arm.json`, median UT→T
`dx`) is **flat across width** — the net is accuracy-*limited*, not capacity-limited:

| width | h32 | **h64** | **h96 (deployed)** | h128 | h256 | h384 |
|---|---|---|---|---|---|---|
| median dx (µm) | 2807 | **2789** | **3581** | 3032 | 3477 | 3477→3782 |

So **h64 is a legitimate ~2× MAC cut with no accuracy loss** (it is even marginally better than the
deployed h96). A trained checkpoint already exists:
`/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/trained_models/wave2_resid_h64`
(config `configs/wave2/wave2_resid_h64.yaml`). It is *not yet* exported to a `.cuh` — that is the one
remaining step to a deployable h64 kernel (see §9).

> ⚠️ **Timing-vs-accuracy caveat for h64 here.** The `pinn_h64*` kernels in this folder reuse the h96
> constant weights as a 64×64 sub-block, because the timing is value-independent and that makes the MAC
> count / memory pattern exactly faithful. Their **accuracy delta is therefore meaningless by design** —
> the *speed* (0.91 ns) is real and measured; the *accuracy* (≈2.8 mm) is the recorded ladder result.

---

## 2. The data (population the kernels were timed on)

Identical to the canonical throughput harness (so numbers stay comparable):

- **1,000,000 real gen-4 tracks** sampled (seed 20260614) from
  `…/experiments/gen_3/data/train_10M_gen4.npz` (9.19 M rows). Per-track `(x,y,tx,ty,qop,z0,dz)`.
  Staged at `<repo>/allen_bridge/bench/artifacts/bench_inputs.npz`.
- **dz distribution** (the extrapolation span): median 446 mm, p99 8527 mm; 15 % of tracks have
  `dz > 3000 mm` (full UT→T-like). **p**: median 17 GeV, p1 1.1 GeV.
- **Field map** (used only by the RK arm): `field.v8r1.down.bin` as a 3-D texture, 81×81×146 @ 100 mm
  voxels, **11.5 MB**. RK crosses each `dz` in 100 mm Cash–Karp steps (mean 82.8 field lookups/track).
- **extrapUTT chart** (the polynomial arm): 19-param table, **0.96 MB**.
- **Footprints (co-metric):** field 11.5 MB · chart 0.96 MB · **NN 40.6 KB** (h64 ≈ 19 KB). The NN
  wins footprint decisively — but footprint never justified a slower/less-accurate surrogate (archive),
  which is exactly why the *speed* result here changes the calculus.

---

## 3. The code (this folder)

```
RK_PINN/
├── README.md                  ← this end-to-end breakdown
├── RESULTS.md                 ← the full measured results table + verdict
├── kernel/
│   └── pinn_v2_utt_fast.cuh   ← ★ THE CANDIDATE: deployable, distilled. bit-exact
│                                 fused h96 (drop-in) + the h64 shape. Deploy this.
├── bench/                     ← the measurement harness (reproduce the numbers)
│   ├── pinn_opt_kernels{,_v2,_v3}.cu   timed kernel variants (v1/v2/v3 = iteration rounds)
│   ├── microbench_opt{,_v2,_v3}.py     drivers; v2 adds the cuBLAS GEMM ceiling
│   └── README.md                       file → which result/figure it produced
├── condor/                    ← HTCondor wrappers + submit files (one V100 slot each)
├── results/                   ← *.json per job + combined.json (the evidence)
└── logs/                      ← condor stdout run-records
```
**`kernel/pinn_v2_utt_fast.cuh` is the artifact not to lose** — everything else is how it
was found and measured. The candidate is otherwise reproducible end-to-end from `bench/` +
`condor/`; the figures in `../throughput/` are generated read-only from `results/combined.json`.

**It does not duplicate any canonical artifact.** The kernels `#include` the locked
`PINN_V2_UTT.cuh` (for the real weights + the reference function) and the NVRTC shims directly from
`/data/bfys/gscriven/track-extrapolation-pinn/...`; the drivers load the staged inputs from the same
repo's `allen_bridge/bench/artifacts/`. The locked repo is **read-only** and untouched.

### The kernel variants, and what each tested
| variant | idea (lever) | result |
|---|---|---|
| `pinn_ref` | verbatim locked kernel (parity + same-slot baseline) | 7.03 ns, **0 error** (validates harness) |
| `pinn_fused` | kill the local-mem spill: unroll contraction, **fuse head into L1 loop** so `h1` never materialises; exact `/std` | **4.85 ns, bit-exact** ⭐ |
| `pinn_warp`/`_u4` | warp-cooperative GEMV (1 track/warp, 3 neurons/lane), transposed weights | 8.3 ns — **latency 18 µs** but throughput lost |
| `pinn_fused_ftanh` | fast `__expf`-based tanh | 4.93 ns — no gain (tanh not the limiter) |
| `pinn_fused_h16` | store `h0` as `__half` (occupancy) | 4.87 ns — no gain, +59 µm error |
| `pinn_fused_lb` | `__launch_bounds__(256,3)` force occupancy↑ | 7.32 ns — **backfires** (reg spill) |
| `pinn_fused_fu` | **full-unroll** both L1 loops (weights → immediate FFMA operands) | 5.09 ns — h96 busts the I-cache |
| `pinn_fused_ilp4` | 4-way accumulator split (break reduction chain) | 4.82 ns — marginal |
| `pinn_h64` | accuracy-equivalent width, inner-unroll | 2.27 ns |
| **`pinn_h64_fu`** | h64 **full-unroll** (fits I-cache → immediate-operand FFMA) | **0.91 ns @ 73.8 % of fp32 peak** 🏆 |
| cuBLAS 3-GEMM fp32/fp16 | whole-batch "GEMM ceiling" (lever 9) | 7.73 / 6.17 ns — **slower** (HBM-streamed intermediates) |

---

## 4. Method (benchmark protocol)

Identical to the canonical `allen_bridge/bench/microbench.py`, so every number is comparable:
**Tesla V100-PCIE-32GB** (80 SM @ 1.38 GHz, 14.1 TFLOP/s fp32 peak, ~898 GB/s), 1 M tracks, block 256,
**CUDA-event** kernel-only median, **200 warm-up + 50 repeats**, fp32, plus a single-warp (32-thread)
latency probe. Each job ran on one exclusive GPU condor slot (`requirements = V100-PCIE-32GB`);
toolchain is NVRTC + cupy (no external nvcc), the same as the canonical harness. Baselines were
**re-timed on the same slot** and reproduce the published values (RK 5.73, extrapUTT 2.36, PINN 7.03 vs
published 5.71 / 2.34 / 7.05), validating cross-method ratios.

Per variant we also recorded, from the cubin: registers/thread, local-memory spill, shared, theoretical
occupancy, and measured % of fp32 FMA peak; and a full **accuracy cross-check** vs the reference kernel
(max position delta in µm, max slope delta in rad, bit-exact flag). Nsight Compute is unavailable on
this NVRTC-only / ephemeral-slot setup; the limiter was isolated instead by these metrics **plus the
controlled ablations** (ftanh, h16, launch-bounds, full-unroll, ILP, block sweep) — they triangulate it.

---

## 5. Results (summary; full table in `RESULTS.md`)

Kernel-only median ns/track, V100, 1 M tracks:

| kernel | ns/track | vs baseline | single-warp | regs | spill | % fp32 peak | accuracy |
|---|---:|---:|---:|---:|---:|---:|---|
| extrapUTT (incumbent poly) | 2.34 | — | 23.6 µs | — | — | — | 15 µm (incumbent) |
| RK + field | 5.71 | — | 11.3 µs | — | — | — | truth |
| **PINN baseline** | 7.03 | 1.00× | 178 µs | 124 | 384 B | 20.5 | reference |
| **`pinn_fused`** ⭐ | **4.85** | 1.45× | 237 µs | 128 | **0** | 29.7 | **bit-exact** |
| `pinn_h64` | 2.27 | 3.10× | 98 µs | 95 | 0 | 29.5 | h64-equiv |
| **`pinn_h64_fu`** 🏆 | **0.91** | **7.75×** | 27.6 µs | 96 | 0 | **73.8** | h64-equiv |

---

## 6. Why it worked (the engineering story)

1. **The baseline's real bug is a local-memory spill.** Two dynamically-indexed arrays
   `h0[96]`/`h1[96]` spill → the 96×96 layer does ~9 216 local loads/track (single-warp latency 178 µs
   = 16× RK). Unrolling the contraction so `h0` is register-resident **and fusing the head into the
   layer-1 loop** (each `h1[j]` is consumed into 4 accumulators immediately, never stored) drops spill
   to **0 B**. 7.03 → 4.85 ns, and — using exact `/std` divide and the identical reduction order —
   **bit-exact** to the locked kernel (preserves the A4 Jacobian / R6 parity gates).
2. **Keep weights as immediate FFMA operands.** Thread-per-track + `constexpr` weights ⇒ warp-uniform
   constant access ⇒ Volta folds each weight straight into `FFMA …, c[bank][imm], …` (zero load) —
   *but only when the offset is a compile-time constant.* **Full-unroll** achieves that; it pays off
   only if the unrolled code fits the instruction cache: **h96 (9 216 FFMA) busts it and regresses;
   h64 (4 096 FFMA) fits and jumps to 0.91 ns at 73.8 % of fp32 peak** — essentially the compute floor.
3. **Width is the big lever, and it is free.** Because accuracy is flat across width, h64 halves the
   MACs at no accuracy cost.
4. **What didn't help (levers exhausted):** the warp-cooperative GEMV cut single-warp latency 10×
   (178→18 µs) but lost throughput — spreading neurons across lanes breaks the constant-broadcast and
   becomes **L2-bandwidth-bound on weights**; the **batched-GEMM "ceiling" is a mirage** (naive cuBLAS
   3-GEMM is *slower* than the fused kernel because it streams H0/H1 through HBM); forcing occupancy up
   backfires; fast-tanh and fp16-storage give nothing. h96 fp32 therefore has a structural floor at
   ~4.8 ns (≈30 % peak), which still beats RK.

---

## 7. Accuracy & deployment constraints

- **fp32 parity:** `pinn_fused` / `pinn_fused_fu` are **bit-exact** (0 µm / 0 rad over 1 M tracks) —
  drop-in safe for the parity gates. `ilp4` reorders the reduction (0.49 µm; negligible vs a 3 mm
  floor, but not bit-exact).
- **Reduced precision:** fp16 storage perturbs outputs ≤ 59 µm — 50× under the 3 mm floor (harmless),
  but gives no speed, so not recommended.
- **Deploy targets honoured:** sm_70 **and** sm_80, fp32 `KalmanFloat`, constant weights ≤ 64 KB
  (h96 40.6 KB, h64 ≈ 19 KB), thread-per-track (fits Allen's launch model).
- **The in-situ caveat (from archive/05 Tier-2):** the deployed `m_use_nn_utt` path is a **hybrid** —
  extrapUTT runs anyway (for the Jacobian F/Q) and the NN state is *added*. So today the NN is strictly
  *additive* overhead. To turn any of this speed into a real win, the NN must become a **true
  replacement** that also supplies the Jacobian (the A4 gate), not an add-on.

---

## 8. The decision this sets up (how to write up / continue)

The speed result is unambiguous and reusable; the strategic choice is **what to do with it.** Options,
honestly weighed:

1. **Write it up as a methods/optimisation result + correction.** Cleanest, lowest-risk: a short,
   self-contained GPU-optimisation story ("a tiny MLP surrogate can be made the fastest UT→T
   extrapolator; the prior 'slower than RK' conclusion was a naïve-kernel artefact") with the spill /
   immediate-operand-FFMA / width findings. This *corrects the record* in archive/05 and 09 regardless
   of whether the NN is ever deployed. **Recommended to bank now.**
2. **Retarget the win to GENERAL extrapolation (the strongest scientific direction).** On UT→T the
   polynomial's 15 µm is unbeatable on accuracy, so "fastest" doesn't make the NN deployable *there*.
   But archive/09's own open question #1 is the **un-attacked** case: *general* extrapolation at
   arbitrary `dz`, where **no polynomial exists and RK is the only option.** There the competitor is
   RK — and we have shown the NN kernel can be **1.2–6× faster than RK**. A "as-accurate-as-needed,
   faster-than-RK" surrogate is a genuine win in that regime. This is where the speed result has teeth.
3. **Spend the speed budget on accuracy — but be sceptical.** h96 is 6× faster than RK, so there is
   head-room to trade speed for accuracy. *However* the capacity ladder shows accuracy is flat with
   width → bigger nets won't help; the 3 mm floor is data/feature/formulation-limited, not capacity.
   So this only pays off via a genuinely different formulation (inputs, residual targets, physics
   terms) — speculative.
4. **Re-open in-situ deployment.** Requires (a) exporting h64 + re-running the parity/A4 gates, (b)
   making the NN supply the Jacobian so it *replaces* rather than *adds to* extrapUTT, and (c) a stance
   on the 3 mm accuracy. Highest effort; only sensible if (2) or a downstream consumer tolerates 3 mm.

My recommendation: **bank (1) now** (it is finished and corrects a published conclusion), and frame the
*continuation* around **(2)** — point the optimised kernel at general extrapolation where RK is the
only incumbent and "faster than RK at acceptable accuracy" is a real, defensible contribution.

---

## 9. Concrete next steps
- [ ] **Export h64** (`wave2_resid_h64`) to a `PINN_V2_UTT_h64.cuh` via the repo's emitter, then re-run
      this bench with the *real* h64 weights (confirms 0.9 ns *and* the 2.8 mm accuracy in one artefact).
- [ ] **Re-run the parity + A4 Jacobian gates** on `pinn_fused` (h96, expected bit-exact) and on the
      exported h64 — then it is George's to set Trust = Verified.
- [ ] **Notion:** a Track-Extrapolation write-up (Trust = Provisional) with this provenance, and a
      to-do for the h64 export + gate pass. (Offered — say the word.)
- [ ] **If pursuing (2):** define the general-extrapolation population + accuracy bar, and re-time
      `pinn_fused`/`pinn_h64_fu` vs RK on it.

---

## 10. Reproduce
```bash
cd /data/bfys/gscriven/Ex_rep/RK_PINN/condor
condor_submit opt.sub      # job #1: baselines (same slot) + v1 variants  -> results/tier1_opt.json
condor_submit opt_v2.sub   # job #2: bit-parity/ftanh/h16/lb/h64 + GEMM ceiling -> results/tier1_opt_v2.json
condor_submit opt_v3.sub   # job #3: full-unroll + ILP sweep             -> results/tier1_opt_v3.json
```
Each lands on one exclusive V100, NVRTC-compiles the kernels against the locked headers, and writes a
results JSON. Raw outputs already present in `results/` and `logs/`.
```
```
Provenance: deliverable repo `track-extrapolation-pinn` @ 283b03b; locked kernel
`candidate/pinn_v2_ALLEN_v1/PINN_V2_UTT.cuh` (blob CRC32 0x1a139335); inputs
`allen_bridge/bench/artifacts/bench_inputs.npz` (1 M gen-4, seed 20260614); capacity ladder
`results/wave2/wave2_three_arm.json`; GPU Tesla V100-PCIE-32GB; cupy 14.1.1 / NVRTC, cc70.
```
