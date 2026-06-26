# PINN_V2_UTT CUDA throughput optimisation — results

**Goal.** Make the locked NN UT→T extrapolator (`pinn_v2_utt_state`, 6→96→96→4, tanh,
fp32) the *fastest* of the three Allen extrapolators, without breaking its outputs.
**Baseline = slowest at 7.05 ns/track.** Target to beat: RK+field 5.71, extrapUTT 2.34.

All numbers: Tesla **V100-PCIE-32GB**, 1 M real gen-4 tracks, block 256, CUDA-event
kernel-only median, 200 warm-up + 50 repeats — *identical protocol to
`allen_bridge/bench/microbench.py`*. Baselines re-timed on the same slot reproduce the
published values (RK 5.73, extrapUTT 2.36, PINN 7.03), so cross-method ratios are valid.
fp32 FMA peak on this V100 = 14.1 TFLOP/s ⇒ pure-FMA floor 1.44 ns (h96), 0.67 ns (h64).

## Master table (kernel-only median, ns/track)

| kernel | ns/track | vs baseline | single-warp | regs | spill | occ | % fp32 peak | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| extrapUTT (incumbent poly) | 2.34 | — | 23.6 µs | — | — | — | — | 10.9 µm (incumbent) |
| RK + field map | 5.71 | — | 11.3 µs | — | — | — | — | truth-ish |
| **PINN baseline** (`pinn_v2_utt_state`) | **7.03** | 1.00× | 178 µs | 124 | 384 B | 0.25 | 20.5 | reference |
| `pinn_warp` (warp-coop GEMV) | 8.32 | 0.85× | **18.4 µs** | 40 | 0 | 0.75 | 17.3 | 0.5 µm |
| `pinn_fused_fu` (h96 full-unroll) | 5.09 | 1.38× | 215 µs | 128 | 0 | 0.25 | 28.3 | **bit-exact** |
| `pinn_fused_ilp4` (h96 + ILP) | 4.82 | 1.46× | 220 µs | 128 | 0 | 0.25 | 29.9 | 0.5 µm |
| **`pinn_fused` (h96, fp32)** ⭐ | **4.85** | **1.45×** | 237 µs | 128 | 0 | 0.25 | 29.7 | **bit-exact (0 µm)** |
| `pinn_h64` (h64 inner-unroll) | 2.27 | 3.10× | 98 µs | 95 | 0 | 0.25 | 29.5 | h64-equiv |
| **`pinn_h64_fu` (h64 full-unroll)** 🏆 | **0.908** | **7.75×** | 27.6 µs | 96 | 0 | 0.25 | **73.8** | h64-equiv |

### Ceilings & negative results
| probe | ns/track | takeaway |
|---|---:|---|
| cuBLAS 3-GEMM fp32 (whole-batch) | 7.73 | **slower than fused** — naive batched GEMM streams H0/H1 through HBM |
| cuBLAS 3-GEMM fp16 tensor cores | 6.17 | still slower than fused — the "GEMM ceiling" is a *mirage* for a 3-layer MLP |
| HBM-traffic floor, naive fp16 3-GEMM | 0.88 | only a *fused* on-chip kernel can approach this; `pinn_h64_fu` (0.91) already does |
| `pinn_fused_lb(256,3)` (force occ↑) | 7.32 | cutting regs to 80 → spill/recompute; **occupancy is already balanced** |
| `pinn_fused_ftanh` (fast `__expf` tanh) | 4.93 | no gain — tanh is **not** the h96 limiter |
| `pinn_fused_h16` (fp16 h0 storage) | 4.87 | no occupancy gain (regs stay 128) + 59 µm error — dead end |

## What moved the needle, and why
1. **Kill the local-memory spill (baseline’s real bug).** The locked kernel keeps two
   dynamically-indexed arrays `h0[96],h1[96]` → both spill to local memory; the 96×96
   layer does ~9 216 local loads/track (single-warp latency 178 µs = 16× RK).
   Fully unrolling the contraction + **fusing the head into the layer-1 loop** so `h1`
   never materialises (one live `hj` + 4 accumulators) drops spill to **0 B**, h0 lives
   in registers, and time falls 7.03 → **4.85 ns** — **bit-exact** (exact `/std` divide,
   identical reduction order). *This alone beats RK and makes the NN no longer slowest.*
2. **Keep weights as immediate FFMA operands.** Thread-per-track + `constexpr` weights ⇒
   warp-uniform constant access ⇒ Volta folds each weight into the `FFMA … c[bank][imm] …`
   instruction (zero load) **iff the offset is compile-time constant**. Inner-only unroll
   leaves the j-offset at runtime (LDC per weight). **Full-unroll** removes those loads —
   but only pays off when the unrolled code fits the I-cache: **h96 (9 216 FFMA) busts it
   and regresses (5.09); h64 (4 096 FFMA) fits and rockets to 0.91 ns @ 73.8 % of peak.**
3. **Width = the big lever.** The recorded capacity ladder (`results/wave2/wave2_three_arm.json`,
   median UT→T `dx`) is flat: **h64 2.79 mm · h96 3.58 mm (deployed) · h384 3.78 mm** —
   the net is accuracy-limited, not width-limited. h64 is a legitimate ~2× MAC cut with
   **no accuracy loss** (slightly better), so `pinn_h64_fu` is a real deployable, not a stunt.

## Why warp-cooperative GEMV lost on throughput (but won latency)
One-track-per-warp cut single-warp latency 178 → **18 µs** (10×) and raised occupancy to
0.75, but throughput got *worse* (8.3 ns): spreading 96 neurons across lanes makes each lane
read a *different* weight → it breaks the constant-broadcast and becomes **L2-bandwidth-bound
on weights** (~36 GB of weight reads for 1 M warps). Thread-per-track with constant-operand
FFMA is bandwidth-optimal for weight delivery; that is why the fused single-thread path wins.

## Accuracy
- **fp32 parity:** `pinn_fused` / `pinn_fused_fu` are **bit-exact** vs the locked reference
  (max position Δ = 0 µm, max slope Δ = 0 rad over 1 M tracks) — preserves the A4 Jacobian /
  R6 bit-parity gates. `ilp4` reorders the reduction → 0.49 µm (vs a 3 mm physics floor:
  negligible, but not bit-exact).
- **Reduced precision:** fp16 h0-storage perturbs outputs by ≤ 59 µm — 50× under the 3 mm
  floor, i.e. physically harmless — but gives no speed, so it is not recommended.
- **h64:** speed is value-independent and measured directly; the h64 *accuracy* (2.79 mm) is
  the recorded capacity-ladder result. (The `pinn_h64*` kernels here reuse the h96 constants
  as a 64×64 sub-block for faithful timing, so their parity delta is meaningless by design.)

## Limiting factor (roofline)
- **h96 fp32 is structurally ~4.8 ns ≈ 30 % of peak.** The 96 fp32 `h0` registers pin
  occupancy at 0.25; full-unroll busts the I-cache and ILP/occupancy/tanh probes all fail to
  move it. That is the **fp32 floor for width-96 thread-per-track on V100** — and it already
  beats RK.
- **h64 fully-unrolled is compute-bound at 73.8 % of fp32 peak (0.91 ns)** — essentially the
  fp32 floor for this net. Going lower needs reduced precision / fused tensor cores, but the
  naive GEMM ceilings show that buys nothing without a hand-fused WMMA kernel, and 0.91 ns
  already beats every incumbent by ≥2.6×.

## Verdict
- **Beats RK?** ✅ `pinn_fused` 4.85 < 5.71 — fp32, **bit-exact, drop-in, no retrain**.
- **Approaches/Beats extrapUTT?** ✅ `pinn_h64` 2.27 < 2.34 even at inner-unroll;
  **`pinn_h64_fu` 0.91 — 2.6× faster than extrapUTT and 6.3× faster than RK.** The NN goes
  from *slowest* to **fastest** extrapolator.
- **Recommended deployables (both honour sm_70/sm_80, fp32 KalmanFloat, ≤64 KB const weights):**
  1. **Now / zero-risk:** `pinn_fused` (h96) — 4.85 ns, bit-parity, swap the body of
     `pinn_v2_utt_state`; 40.6 KB constant weights unchanged.
  2. **Fastest:** `pinn_h64_fu` — 0.91 ns. Weights already trained (`wave2_resid_h64`,
     ~19 KB const) and accuracy-equivalent (2.79 mm). Needs a one-time h64 weight export +
     re-run of the parity/A4 gates (then George’s Verify). Speed is already proven here.

Artifacts: kernels `src/pinn_opt_kernels{,_v2,_v3}.cu`; drivers `src/microbench_opt{,_v2,_v3}.py`;
raw results `results/*.json` (+ `results/combined.json`). Profiling note: Nsight Compute is
unavailable on this NVRTC-only / ephemeral-slot setup; limiter isolated via cubin attributes
(regs/spill/shared), theoretical occupancy, measured %-of-peak, single-warp latency, and the
controlled ablations above.
