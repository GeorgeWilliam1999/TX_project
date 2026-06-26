#!/usr/bin/env python3
"""microbench_opt_v2.py — round-2 PINN optimisation bench + GEMM ceiling.

Same protocol as microbench.py (200 warmup, 50 repeats, block 256, CUDA events,
single-warp probe). Adds: per-channel accuracy vs reference reported as a PHYSICAL
position delta (um) against the ~3 mm net-error floor; a block-size sweep for the
winning kernel; and the whole-batch cuBLAS GEMM ceiling (fp32 and fp16 tensor
cores) with the intermediate-activation HBM-traffic floor.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
BENCH = os.path.join(REPO, "allen_bridge", "bench")
MACS = {"h96": 6*96+96*96+96*4, "h64": 6*64+64*64+64*4}


def gpu_info():
    import subprocess
    try:
        q = "name,driver_version,clocks.max.sm,memory.total"
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).strip()
        return dict(zip(q.split(","), [s.strip() for s in out.splitlines()[0].split(",")]))
    except Exception as e:
        return {"nvidia_smi_error": str(e)}


def pct(a):
    a = np.asarray(a, np.float64); p = np.percentile(a, [0, 25, 50, 75, 100])
    return {"median": float(p[2]), "p25": float(p[1]), "p75": float(p[3]),
            "min": float(p[0]), "max": float(p[4]),
            "rel_iqr": float((p[3]-p[1])/p[2]) if p[2] else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.path.join(BENCH, "artifacts"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "tier1_opt_v2.json"))
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
    h = {k: np.ascontiguousarray(inp[k], dtype=np.float32) for k in ["x","y","tx","ty","qop","z0","dz"]}
    N = h["x"].shape[0]
    dev = cp.cuda.Device(); cc = dev.compute_capability
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    sm = props["multiProcessorCount"]; clk = props["clockRate"]
    regs_sm = props["regsPerMultiprocessor"]; maxthr = props["maxThreadsPerMultiProcessor"]
    peak = sm * 64 * 2 * (clk * 1e3)
    mem_bw = props.get("memoryClockRate", 877000)*1e3 * (props.get("memoryBusWidth",4096)/8) * 2  # bytes/s

    incs = [os.path.join(BENCH, "shims"), os.path.join(REPO, "candidate", "pinn_v2_ALLEN_v1"), HERE]
    options = tuple(["--std=c++20"] + [f"-I{d}" for d in incs])
    with open(os.path.join(HERE, "pinn_opt_kernels_v2.cu")) as f:
        src = f.read()
    t0 = time.time()
    module = cp.RawModule(code=src, backend="nvrtc", options=options)
    names = ["pinn_ref","pinn_fused","pinn_fused_ftanh","pinn_fused_h16","pinn_fused_lb","pinn_h64"]
    fns = {n: module.get_function(n) for n in names}
    print(f"compiled {time.time()-t0:.2f}s cc{cc} {sm}SM @ {clk/1e6:.3f}GHz peak={peak/1e12:.1f}TF mem_bw~{mem_bw/1e9:.0f}GB/s")

    d = {k: cp.asarray(v) for k, v in h.items()}
    ref = [cp.empty(N, np.float32) for _ in range(4)]
    out = [cp.empty(N, np.float32) for _ in range(4)]

    def mkargs(n, o): return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], d["dz"], np.int32(n), *o)
    def launch(fn, n, block, o):
        grid = ((n + block - 1)//block,); fn(grid, (block,), mkargs(n, o))

    def attrs(fn, block):
        a = {}
        for k in ("num_regs","local_size_bytes","shared_size_bytes","max_threads_per_block"):
            try: a[k] = int(getattr(fn, k))
            except Exception: a[k] = None
        try:
            regs = a["num_regs"] or 1
            rpw = ((regs*32 + 255)//256)*256
            blocks = max(1, min(regs_sm//(rpw*(block//32)), maxthr//block, 32))
            a["theoretical_occupancy"] = min(1.0, blocks*block/maxthr)
        except Exception: a["theoretical_occupancy"] = None
        return a

    launch(fns["pinn_ref"], N, args.block, ref); dev.synchronize()
    refh = [cp.asnumpy(r) for r in ref]

    def accuracy(fn):
        for b in out: b[:] = 0
        launch(fn, N, args.block, out); dev.synchronize()
        oh = [cp.asnumpy(o) for o in out]
        finite = all(bool(np.all(np.isfinite(o))) for o in oh)
        # x,y are mm -> report um; tx,ty dimensionless (rad)
        dpos_um = max(float(np.abs(oh[0]-refh[0]).max()), float(np.abs(oh[1]-refh[1]).max())) * 1e3
        dpos_med_um = max(float(np.median(np.abs(oh[0]-refh[0]))), float(np.median(np.abs(oh[1]-refh[1])))) * 1e3
        dslope = max(float(np.abs(oh[2]-refh[2]).max()), float(np.abs(oh[3]-refh[3]).max()))
        # max rel over all channels (for the strict fp32 parity gate)
        mrel = 0.0
        for k in range(4):
            mrel = max(mrel, float((np.abs(oh[k]-refh[k])/(np.abs(refh[k])+1e-6)).max()))
        return {"finite": finite, "max_pos_delta_um": dpos_um, "median_pos_delta_um": dpos_med_um,
                "max_slope_delta_rad": dslope, "max_rel_err": mrel,
                "bit_exact": dpos_um == 0.0 and dslope == 0.0}

    def time_only(fn, n, block):
        for _ in range(args.warmup): launch(fn, n, block, out)
        dev.synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event(); ts = []
        for _ in range(args.repeats):
            e0.record(); launch(fn, n, block, out); e1.record(); e1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(e0, e1))
        return ts

    results = {}
    for key in names:
        fn = fns[key]; macs = MACS["h64"] if key == "pinn_h64" else MACS["h96"]
        acc = accuracy(fn)
        ko = pct(time_only(fn, N, args.block))
        warp = pct(time_only(fn, 32, 32))
        ns = ko["median"]*1e6/N
        a = attrs(fn, args.block)
        results[key] = {"ns_per_track": ns, "kernel_only_ms": ko,
                        "single_warp_us": warp["median"]*1000,
                        "pct_fp32_peak": 100.0*(2*macs/(ns*1e-9))/peak,
                        "macs_per_track": macs, "accuracy_vs_ref": acc, "resources": a}
        print(f"{key:18s} {ns:7.3f} ns  warp={warp['median']*1000:7.1f}us regs={a['num_regs']:3d} "
              f"occ={a['theoretical_occupancy']} %pk={results[key]['pct_fp32_peak']:5.1f} "
              f"dpos={acc['max_pos_delta_um']:.3g}um dslope={acc['max_slope_delta_rad']:.2e} "
              f"bitexact={acc['bit_exact']}")

    # ---- block-size sweep for the winning bit-parity kernel ----
    sweep = {}
    for blk in (64, 128, 192, 256, 384, 512):
        ko = pct(time_only(fns["pinn_fused"], N, blk))
        sweep[blk] = ko["median"]*1e6/N
    print("pinn_fused block sweep ns/track:", {k: round(v,3) for k,v in sweep.items()})

    # ---- whole-batch cuBLAS GEMM ceiling (fp32 + fp16 tensor cores) ----
    rng = np.random.default_rng(0)
    Xn = np.stack([(h["x"]-(-1.3))/2011, (h["y"]-0.77)/1440, h["tx"]/0.23, h["ty"]/0.20,
                   h["qop"]/0.092, np.ones(N, np.float32)], axis=1).astype(np.float32)  # [N,6] (values ~representative)
    W0 = rng.standard_normal((96,6)).astype(np.float32)*0.1; b0 = rng.standard_normal(96).astype(np.float32)*0.01
    W1 = rng.standard_normal((96,96)).astype(np.float32)*0.1; b1 = rng.standard_normal(96).astype(np.float32)*0.01
    W2 = rng.standard_normal((4,96)).astype(np.float32)*0.1; b2 = rng.standard_normal(4).astype(np.float32)*0.01

    def gemm_ceiling(dtype, label):
        Xg = cp.asarray(Xn.astype(dtype)); W0g = cp.asarray(W0.T.astype(dtype)); b0g = cp.asarray(b0.astype(dtype))
        W1g = cp.asarray(W1.T.astype(dtype)); b1g = cp.asarray(b1.astype(dtype))
        W2g = cp.asarray(W2.T.astype(dtype)); b2g = cp.asarray(b2.astype(dtype))
        def run():
            H0 = cp.tanh(Xg @ W0g + b0g); H1 = cp.tanh(H0 @ W1g + b1g); C = H1 @ W2g + b2g
            return C
        for _ in range(30): run()
        dev.synchronize(); e0,e1 = cp.cuda.Event(), cp.cuda.Event(); ts=[]
        for _ in range(args.repeats):
            e0.record(); run(); e1.record(); e1.synchronize(); ts.append(cp.cuda.get_elapsed_time(e0,e1))
        p = pct(ts); ns = p["median"]*1e6/N
        # HBM traffic floor for the naive 3-GEMM (intermediates streamed to/from HBM)
        b = np.dtype(dtype).itemsize
        bytes_per_track = (6 + 96 + 96 + 96 + 96 + 4) * b  # read X, w/r H0, w/r H1, write C (approx)
        hbm_floor_ns = bytes_per_track / mem_bw * 1e9
        print(f"GEMM ceiling {label:5s}: {ns:7.3f} ns/track  (HBM-traffic floor ~{hbm_floor_ns:.3f} ns/track)")
        return {"ns_per_track": ns, "kernel_only_ms": p, "hbm_traffic_floor_ns_per_track": hbm_floor_ns,
                "note": "3 separate cuBLAS GEMMs; intermediate activations streamed through HBM"}

    gemm = {"fp32": gemm_ceiling(np.float32, "fp32"), "fp16": gemm_ceiling(np.float16, "fp16")}

    base = results["pinn_ref"]["ns_per_track"]
    blob = {"tier":1, "variant_set":"v2", "n_tracks":int(N), "block_size":args.block,
            "warmup_iters":args.warmup, "timed_repeats":args.repeats,
            "published_baselines_ns_per_track":{"extrapUTT":2.344,"rk_field":5.709,"pinn_v2_baseline":7.054},
            "physics_floor_um":{"h96_deployed":3580.8,"h64":2788.6,"h384":3782.3,
                                "source":"results/wave2/wave2_three_arm.json (median_dx_um)"},
            "toolchain":{"compiler":"NVRTC","cupy":cp.__version__,"cc":cc,"nvrtc_options":list(options)},
            "gpu":gpu_info(), "device_props":{"sm":sm,"clock_ghz":clk/1e6,"fp32_peak_tflop":peak/1e12,
                                              "mem_bw_GBs":mem_bw/1e9},
            "methods":results, "speedup_vs_ref":{k:base/results[k]["ns_per_track"] for k in results},
            "pinn_fused_block_sweep_ns":sweep, "gemm_ceiling":gemm}
    with open(args.out, "w") as f: json.dump(blob, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
