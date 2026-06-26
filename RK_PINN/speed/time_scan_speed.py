#!/usr/bin/env python3
"""time_scan_speed.py — V100 throughput exam for the scan architectures, SAME protocol
as the last round (1M gen-4 tracks, block sweep, 200 warmup / 50 repeat, CUDA events).

Each scan_kernels/<label>.cu is compiled as its OWN NVRTC module (own ≤64 KB constant
bank) and both variants (_fused, _fu) are timed across blocks {64..384}; we report the
MIN ns/track per shape×activation — exactly how the last exam took pinn_fused (h96) and
pinn_h64_fu (h64). Anchors (h96_tanh, d2_tanh) must reproduce ~4.85 / ~0.91 ns.

Writes results/scan_speed.json.
"""
import glob, json, os, sys, time
import numpy as np

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
BENCH = os.path.join(REPO, "allen_bridge", "bench")
HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "scan_kernels")
BLOCKS = (64, 96, 128, 192, 256, 384)


def pct(a):
    a = np.asarray(a, np.float64); p = np.percentile(a, [25, 50, 75])
    return {"median": float(p[1]), "rel_iqr": float((p[2]-p[0])/p[1]) if p[1] else None}


def main():
    sys.path.insert(0, os.path.join(BENCH, "_pyenv"))
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(HERE, ".cupy_cache"))
    os.makedirs(os.environ["CUPY_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(HERE), "results"), exist_ok=True)
    import cupy as cp

    inp = np.load(os.path.join(BENCH, "artifacts", "bench_inputs.npz"))
    h = {k.upper(): np.ascontiguousarray(inp[k], np.float32) for k in ["x","y","tx","ty","qop","dz"]}
    N = h["X"].shape[0]
    d = {k: cp.asarray(v) for k, v in h.items()}
    OX, OY, OTX, OTY = (cp.empty(N, np.float32) for _ in range(4))
    args = (d["X"], d["Y"], d["TX"], d["TY"], d["QOP"], d["DZ"], np.int32(N), OX, OY, OTX, OTY)
    dev = cp.cuda.Device(); cc = dev.compute_capability
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(f"host {os.uname().nodename}  cc{cc} ({name})  N={N}")

    manifest = json.load(open(os.path.join(KDIR, "manifest.json")))

    def time_fn(fn, block, warmup=200, repeats=50):
        grid = ((N + block - 1) // block,)
        for _ in range(warmup):
            fn(grid, (block,), args)
        dev.synchronize(); e0, e1 = cp.cuda.Event(), cp.cuda.Event(); ts = []
        for _ in range(repeats):
            e0.record(); fn(grid, (block,), args); e1.record(); e1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(e0, e1))
        return pct(ts)["median"] * 1e6 / N  # ns/track

    res = {}
    for cu in sorted(glob.glob(os.path.join(KDIR, "*.cu"))):
        label = os.path.splitext(os.path.basename(cu))[0]
        src = open(cu).read()
        t0 = time.time()
        mod = cp.RawModule(code=src, backend="nvrtc", options=("--std=c++17",))
        best = {"ns": 1e9, "variant": None, "block": None}; per = {}
        for variant in ("fused", "fu"):
            try:
                fn = mod.get_function(f"{label}_{variant}")
            except Exception as e:
                print(f"  {label}_{variant}: MISSING ({str(e)[:40]})"); continue
            per[variant] = {}
            for blk in BLOCKS:
                ns = time_fn(fn, blk); per[variant][blk] = ns
                if ns < best["ns"]:
                    best = {"ns": ns, "variant": variant, "block": blk}
        m = manifest[label]
        res[label] = {"dims": m["dims"], "act": m["act"], "macs": m["macs"],
                      "n_act": m["n_act"], "weights": m["weights"],
                      "best_ns_per_track": best["ns"], "best_variant": best["variant"],
                      "best_block": best["block"], "sweep": per}
        print(f"{label:10s} {str(m['dims']):14s} {m['act']:5s} "
              f"-> {best['ns']:.3f} ns/track  ({best['variant']} blk{best['block']})  "
              f"[{m['weights']}]  compile {time.time()-t0:.1f}s")

    out = {"hw": name, "cc": cc, "n_tracks": int(N), "warmup": 200, "repeats": 50,
           "protocol": "same as throughput bench (CUDA events, block sweep, min over variants)",
           "anchors_published": {"h96_tanh": 4.85, "d2_tanh(h64)": 0.91, "extrapUTT": 2.34,
                                 "nystrom_fast": 0.405, "rk_cashkarp": 5.63},
           "kernels": res}
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "results", "scan_speed.json"), "w"), indent=2)
    print("\nwrote results/scan_speed.json")
    # anchor check
    for a, exp in (("h96_tanh", 4.85), ("d2_tanh", 0.91)):
        if a in res:
            got = res[a]["best_ns_per_track"]
            print(f"ANCHOR {a}: got {got:.3f} ns vs published {exp} ns "
                  f"({'OK' if abs(got-exp)/exp < 0.25 else 'CHECK'})")


if __name__ == "__main__":
    main()
