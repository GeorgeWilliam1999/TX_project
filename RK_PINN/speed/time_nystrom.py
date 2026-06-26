#!/usr/bin/env python3
"""time_nystrom.py — GPU timing of Allen's fast Nystrom make_fast_step vs Cash-Karp RK,
same V100 protocol/population/field as the throughput bench (200 warmup, 50 repeats,
CUDA events, block 256, 1M gen-4 tracks). Writes results/nystrom_speed.json."""
from __future__ import annotations
import json, os, sys, time
import numpy as np

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
BENCH = os.path.join(REPO, "allen_bridge", "bench")
HERE = os.path.dirname(os.path.abspath(__file__))
ALLEN = os.environ.get("ALLEN_DIR", "/data/bfys/gscriven/Allen")


def pct(a):
    a = np.asarray(a, np.float64); p = np.percentile(a, [25, 50, 75])
    return {"median": float(p[1]), "rel_iqr": float((p[2]-p[0])/p[1]) if p[1] else None}


def main():
    sys.path.insert(0, os.path.join(BENCH, "_pyenv"))
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(HERE, ".cupy_cache"))
    os.makedirs(os.environ["CUPY_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(HERE), "results"), exist_ok=True)
    import cupy as cp
    from cupy.cuda import texture, runtime

    inp = np.load(os.path.join(BENCH, "artifacts", "bench_inputs.npz"))
    h = {k: np.ascontiguousarray(inp[k], np.float32) for k in ["x","y","tx","ty","qop","z0","dz"]}
    N = h["x"].shape[0]
    fld = np.load(os.path.join(BENCH, "artifacts", "field_v8r1_down.npz"))
    Bx, By, Bz = fld["Bx"], fld["By"], fld["Bz"]
    invD, mn, Ng = fld["invD"], fld["minXYZ"], fld["N"]
    nx, ny, nz = (int(v) for v in Ng)
    dev = cp.cuda.Device(); cc = dev.compute_capability
    props = cp.cuda.runtime.getDeviceProperties(dev.id)

    incs = [os.path.join(BENCH, "shims"), BENCH, HERE,
            os.path.join(ALLEN, "device", "kalman", "ParKalman", "include"),
            os.path.join(ALLEN, "device", "event_model", "common", "include")]
    opts = tuple(["--std=c++20", "-DMAGFIELD_USE_TEXTURE"] + [f"-I{d}" for d in incs])
    with open(os.path.join(HERE, "nystrom_speed.cu")) as f:
        src = f.read()
    t0 = time.time()
    mod = cp.RawModule(code=src, backend="nvrtc", options=opts)
    rk = mod.get_function("rk_kernel"); nys = mod.get_function("nystrom_kernel")
    print(f"compiled in {time.time()-t0:.1f}s on cc{cc} ({props['name'].decode() if isinstance(props['name'],bytes) else props['name']})")

    d = {k: cp.asarray(v) for k, v in h.items()}
    ox, oy, otx, oty = (cp.empty(N, np.float32) for _ in range(4))
    ch = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuarrs, texobjs = [], []
    for comp in (Bx, By, Bz):
        arr = texture.CUDAarray(ch, nx, ny, nz); arr.copy_from(cp.asarray(comp))
        res = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        td = texture.TextureDescriptor((runtime.cudaAddressModeClamp,)*3, runtime.cudaFilterModeLinear,
                                       runtime.cudaReadModeElementType, normalizedCoords=0)
        texobjs.append(texture.TextureObject(res, td)); cuarrs.append(arr)
    tBx, tBy, tBz = texobjs
    f32 = np.float32
    minX, minY, minZ = (f32(v) for v in mn); iDx, iDy, iDz = (f32(v) for v in invD)
    MAXS = np.int32(200)

    def args(step):
        return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], d["z0"], d["dz"], np.int32(N),
                f32(step), MAXS, tBx, tBy, tBz, minX, minY, minZ, iDx, iDy, iDz, ox, oy, otx, oty)

    block = 256; grid = ((N + block - 1)//block,)
    def time_k(fn, step, warmup=200, repeats=50):
        a = args(step)
        for _ in range(warmup): fn(grid, (block,), a)
        dev.synchronize(); e0, e1 = cp.cuda.Event(), cp.cuda.Event(); ts = []
        for _ in range(repeats):
            e0.record(); fn(grid, (block,), a); e1.record(); e1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(e0, e1))
        return ts

    res = {}
    for name, fn, step in [("rk_field_cashkarp_100mm", rk, 100.0), ("nystrom_fast_500mm", nys, 500.0)]:
        p = pct(time_k(fn, step)); ns = p["median"]*1e6/N
        res[name] = {"ns_per_track": ns, "median_ms": p["median"], "rel_iqr": p["rel_iqr"], "step_mm": step}
        print(f"{name:28s} {ns:7.3f} ns/track  rel_iqr={p['rel_iqr']*100:.2f}%")

    out = {"hw": "V100 (same protocol as throughput bench)", "cc": cc, "n_tracks": int(N),
           "block": block, "warmup": 200, "repeats": 50,
           "published_same_protocol": {"extrapUTT": 2.34, "NN_baseline": 7.05, "NN_fused_h96": 4.85, "NN_h64_fu": 0.91},
           "methods": res}
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "results", "nystrom_speed.json"), "w"), indent=2)
    print("wrote results/nystrom_speed.json")


if __name__ == "__main__":
    main()
