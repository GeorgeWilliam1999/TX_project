#!/usr/bin/env python3
"""microbench_opt.py — Tier-1 throughput bench for OPTIMISED PINN_V2_UTT variants.

Same protocol as allen_bridge/bench/microbench.py: CUDA-event timing, >=200
warm-up iters discarded, >=50 timed repeats, median+IQR, block size 256, plus a
single-warp (32-thread) latency probe. Adds:
  - accuracy cross-check of every variant vs the verbatim reference kernel
    (max abs / max rel error over all 4 outputs), and
  - per-kernel resource stats (regs/thread, local spill, shared, occupancy,
    % of V100 fp32 FMA peak) from the cuBin.

Variants timed: pinn_ref (locked, same-slot baseline), pinn_fused (thread/track,
register-resident, fused head), pinn_warp / pinn_warp_u4 (warp-cooperative GEMV).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
BENCH = os.path.join(REPO, "allen_bridge", "bench")

# V100 fp32: 80 SM * 64 FP32 cores * 2 (FMA) * boost clock. Peak FLOP/s used only
# for the "% of peak" figure; we report MACs/track too so the reader can recompute.
MACS_PER_TRACK = 6 * 96 + 96 * 96 + 96 * 4          # 10176
FLOP_PER_TRACK = 2 * MACS_PER_TRACK                  # count FMA as 2 flop


def gpu_info():
    import subprocess
    try:
        q = "name,driver_version,clocks.max.sm,clocks.sm,memory.total"
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).strip()
        return dict(zip(q.split(","), [s.strip() for s in out.splitlines()[0].split(",")]))
    except Exception as e:
        return {"nvidia_smi_error": str(e)}


def percentiles(a):
    a = np.asarray(a, dtype=np.float64)
    p = np.percentile(a, [0, 25, 50, 75, 100])
    med = float(p[2]); iqr = float(p[3] - p[1])
    return {"min": float(p[0]), "p25": float(p[1]), "median": med, "p75": float(p[3]),
            "max": float(p[4]), "iqr": iqr, "rel_iqr": float(iqr / med) if med else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.path.join(BENCH, "artifacts"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "tier1_opt.json"))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--pyenv", default=os.path.join(BENCH, "_pyenv"))
    args = ap.parse_args()

    if args.pyenv and os.path.isdir(args.pyenv):
        sys.path.insert(0, args.pyenv)
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(HERE, ".cupy_cache"))
    os.makedirs(os.environ["CUPY_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import cupy as cp

    inp = np.load(os.path.join(args.art, "bench_inputs.npz"))
    h = {k: np.ascontiguousarray(inp[k], dtype=np.float32) for k in
         ["x", "y", "tx", "ty", "qop", "z0", "dz"]}
    N = h["x"].shape[0]
    dev = cp.cuda.Device()
    cc = dev.compute_capability
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    sm_count = props["multiProcessorCount"]
    max_thr_sm = props["maxThreadsPerMultiProcessor"]
    regs_per_sm = props["regsPerMultiprocessor"]
    clock_khz = props["clockRate"]                  # boost clock, kHz
    fp32_peak_flop = sm_count * 64 * 2 * (clock_khz * 1e3)

    incs = [os.path.join(BENCH, "shims"), os.path.join(REPO, "candidate", "pinn_v2_ALLEN_v1"), HERE]
    options = tuple(["--std=c++20"] + [f"-I{d}" for d in incs])
    with open(os.path.join(HERE, "pinn_opt_kernels.cu")) as f:
        src = f.read()
    t0 = time.time()
    module = cp.RawModule(code=src, backend="nvrtc", options=options)
    fns = {name: module.get_function(name) for name in
           ["pinn_ref", "pinn_fused", "pinn_warp", "pinn_warp_u4", "init_weights"]}
    compile_s = time.time() - t0
    print(f"compiled in {compile_s:.2f}s on cc{cc}, {sm_count} SM @ {clock_khz/1e6:.3f} GHz "
          f"=> fp32 peak {fp32_peak_flop/1e12:.1f} TFLOP/s")

    d = {k: cp.asarray(v) for k, v in h.items()}
    # reference outputs (computed once) + per-variant output buffers
    ref = [cp.empty(N, np.float32) for _ in range(4)]
    out = [cp.empty(N, np.float32) for _ in range(4)]

    # transposed weight buffers for warp variants
    Wt0 = cp.empty(6 * 96, np.float32); Wt1 = cp.empty(96 * 96, np.float32)
    W2c = cp.empty(4 * 96, np.float32)
    B0 = cp.empty(96, np.float32); B1 = cp.empty(96, np.float32); B2 = cp.empty(4, np.float32)
    fns["init_weights"]((64,), (256,), (Wt0, Wt1, W2c, B0, B1, B2))
    dev.synchronize()

    def args_tpt(n, o):   # thread-per-track variants
        return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], d["dz"], np.int32(n), *o)

    def args_warp(n, o):  # warp-cooperative variants
        return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], d["dz"], np.int32(n),
                Wt0, Wt1, W2c, B0, B1, B2, *o)

    # (function, arg-builder, threads-per-track)
    VARIANTS = {
        "pinn_ref":     (fns["pinn_ref"],    args_tpt,  1),
        "pinn_fused":   (fns["pinn_fused"],  args_tpt,  1),
        "pinn_warp":    (fns["pinn_warp"],   args_warp, 32),
        "pinn_warp_u4": (fns["pinn_warp_u4"],args_warp, 32),
    }

    def launch(fn, mkargs, tpt, n, block, o):
        threads = n * tpt
        grid = ((threads + block - 1) // block,)
        smem = 2 * (block // 32) * 96 * 4 if tpt == 32 else 0
        fn(grid, (block,), mkargs(n, o), shared_mem=smem)

    def attrs(fn, block, tpt):
        a = {}
        for k in ("num_regs", "local_size_bytes", "shared_size_bytes",
                  "const_size_bytes", "max_threads_per_block"):
            try: a[k] = int(getattr(fn, k))
            except Exception: a[k] = None
        # register-limited theoretical occupancy at this block size
        smem = 2 * (block // 32) * 96 * 4 if tpt == 32 else 0
        try:
            regs = a["num_regs"] or 1
            regs_per_block = ((regs * 32 + 255) // 256) * 256 * (block // 32)  # 256-reg granularity/warp
            blk_reg = regs_per_sm // max(regs_per_block, 1)
            blk_thr = max_thr_sm // block
            blk_smem = (98304 // smem) if smem else 999
            blocks = max(1, min(blk_reg, blk_thr, blk_smem, 32))
            a["theoretical_occupancy"] = min(1.0, blocks * block / max_thr_sm)
        except Exception:
            a["theoretical_occupancy"] = None
        return a

    # ----- reference outputs -----
    launch(fns["pinn_ref"], args_tpt, 1, N, args.block, ref)
    dev.synchronize()
    ref_h = [cp.asnumpy(r) for r in ref]

    def accuracy(fn, mkargs, tpt):
        for b in out: b[:] = 0
        launch(fn, mkargs, tpt, N, args.block, out)
        dev.synchronize()
        max_abs = 0.0; max_rel = 0.0; finite = True
        for k in range(4):
            o = cp.asnumpy(out[k]); r = ref_h[k]
            finite = finite and bool(np.all(np.isfinite(o)))
            ae = np.abs(o - r)
            re = ae / (np.abs(r) + 1e-6)
            max_abs = max(max_abs, float(ae.max())); max_rel = max(max_rel, float(re.max()))
        return {"max_abs_err": max_abs, "max_rel_err": max_rel, "finite": finite}

    def time_kernel_only(fn, mkargs, tpt, n, block):
        for _ in range(args.warmup):
            launch(fn, mkargs, tpt, n, block, out)
        dev.synchronize()
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ts = []
        for _ in range(args.repeats):
            ev0.record(); launch(fn, mkargs, tpt, n, block, out); ev1.record(); ev1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(ev0, ev1))  # ms
        return ts

    results = {}
    for key, (fn, mkargs, tpt) in VARIANTS.items():
        acc = accuracy(fn, mkargs, tpt)
        ko = time_kernel_only(fn, mkargs, tpt, N, args.block)
        # single-warp latency: one warp (32 threads). thread/track => 32 tracks;
        # warp-coop => 1 track. block=32.
        nwarp = 32 // tpt if tpt <= 32 else 1
        warp = time_kernel_only(fn, mkargs, tpt, max(nwarp, 1), 32)
        kp = percentiles(ko); wp = percentiles(warp)
        ns_per_track = kp["median"] * 1e6 / N
        a = attrs(fn, args.block, tpt)
        achieved_flop = FLOP_PER_TRACK / (ns_per_track * 1e-9)
        results[key] = {
            "kernel_only_ms": kp,
            "single_warp_us": {k: (v * 1000 if isinstance(v, float) else v) for k, v in wp.items()},
            "single_warp_tracks": max(nwarp, 1),
            "ns_per_track": ns_per_track,
            "tracks_per_s": N / (kp["median"] * 1e-3),
            "pct_fp32_peak": 100.0 * achieved_flop / fp32_peak_flop,
            "accuracy_vs_ref": acc,
            "resources": a,
        }
        print(f"{key:14s} {ns_per_track:7.3f} ns/track  warp={wp['median']*1000:8.1f}us "
              f"regs={a['num_regs']} spill={a['local_size_bytes']}B "
              f"occ={a['theoretical_occupancy']} %peak={results[key]['pct_fp32_peak']:.1f} "
              f"max_rel={acc['max_rel_err']:.2e} max_abs={acc['max_abs_err']:.2e}")

    base = results["pinn_ref"]["ns_per_track"]
    summary = {k: {"ns_per_track": results[k]["ns_per_track"],
                   "speedup_vs_ref": base / results[k]["ns_per_track"]} for k in results}
    blob = {
        "tier": 1, "variant_set": "fp32_inkernel_v1", "n_tracks": int(N),
        "block_size": args.block, "warmup_iters": args.warmup, "timed_repeats": args.repeats,
        "macs_per_track": MACS_PER_TRACK, "dtype": "fp32",
        "published_baselines_ns_per_track": {"extrapUTT": 2.344, "rk_field": 5.709, "pinn_v2_baseline": 7.054},
        "toolchain": {"compiler": "NVRTC", "cupy": cp.__version__, "cc": cc,
                      "nvrtc_options": list(options), "compile_seconds": round(compile_s, 2)},
        "gpu": gpu_info(),
        "device_props": {"sm_count": sm_count, "clock_ghz": clock_khz / 1e6,
                         "fp32_peak_tflop": fp32_peak_flop / 1e12},
        "methods": results, "summary_vs_ref": summary,
    }
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
