#!/usr/bin/env python3
"""microbench_opt_v3.py — full-unroll / ILP lever sweep on the throughput winner."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
BENCH = os.path.join(REPO, "allen_bridge", "bench")
MACS = {"pinn_h64": 6*64+64*64+64*4, "pinn_h64_fu": 6*64+64*64+64*4}
DEF_MAC = 6*96+96*96+96*4


def pct(a):
    a = np.asarray(a, np.float64); p = np.percentile(a, [25, 50, 75])
    return {"median": float(p[1]), "rel_iqr": float((p[2]-p[0])/p[1]) if p[1] else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "tier1_opt_v3.json"))
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--pyenv", default=os.path.join(BENCH, "_pyenv"))
    args = ap.parse_args()
    if args.pyenv and os.path.isdir(args.pyenv): sys.path.insert(0, args.pyenv)
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(HERE, ".cupy_cache"))
    os.makedirs(os.environ["CUPY_CACHE_DIR"], exist_ok=True); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import cupy as cp

    inp = np.load(os.path.join(BENCH, "artifacts", "bench_inputs.npz"))
    h = {k: np.ascontiguousarray(inp[k], np.float32) for k in ["x","y","tx","ty","qop","dz"]}
    N = h["x"].shape[0]
    dev = cp.cuda.Device(); cc = dev.compute_capability
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    sm = props["multiProcessorCount"]; clk = props["clockRate"]
    regs_sm = props["regsPerMultiprocessor"]; maxthr = props["maxThreadsPerMultiProcessor"]
    peak = sm*64*2*(clk*1e3)

    incs = [os.path.join(BENCH,"shims"), os.path.join(REPO,"candidate","pinn_v2_ALLEN_v1"), HERE]
    options = tuple(["--std=c++20"]+[f"-I{x}" for x in incs])
    with open(os.path.join(HERE, "pinn_opt_kernels_v3.cu")) as f: src = f.read()
    t0 = time.time(); module = cp.RawModule(code=src, backend="nvrtc", options=options)
    names = ["pinn_ref","pinn_fused","pinn_fused_fu","pinn_fused_ilp4","pinn_h64","pinn_h64_fu"]
    fns = {n: module.get_function(n) for n in names}
    print(f"compiled {time.time()-t0:.2f}s cc{cc} {sm}SM @ {clk/1e6:.3f}GHz peak={peak/1e12:.1f}TF")

    d = {k: cp.asarray(v) for k, v in h.items()}
    ref = [cp.empty(N, np.float32) for _ in range(4)]; out = [cp.empty(N, np.float32) for _ in range(4)]
    def mkargs(n,o): return (d["x"],d["y"],d["tx"],d["ty"],d["qop"],d["dz"],np.int32(n),*o)
    def launch(fn,n,blk,o): fn(((n+blk-1)//blk,),(blk,),mkargs(n,o))
    def attrs(fn,blk):
        a={}
        for k in ("num_regs","local_size_bytes","shared_size_bytes"):
            try: a[k]=int(getattr(fn,k))
            except Exception: a[k]=None
        try:
            rpw=((a["num_regs"]*32+255)//256)*256
            blocks=max(1,min(regs_sm//(rpw*(blk//32)),maxthr//blk,32)); a["occ"]=min(1.0,blocks*blk/maxthr)
        except Exception: a["occ"]=None
        return a
    launch(fns["pinn_ref"],N,args.block,ref); dev.synchronize(); refh=[cp.asnumpy(r) for r in ref]
    def accuracy(fn):
        for b in out: b[:]=0
        launch(fn,N,args.block,out); dev.synchronize(); oh=[cp.asnumpy(o) for o in out]
        finite=all(bool(np.all(np.isfinite(o))) for o in oh)
        dpos=max(float(np.abs(oh[0]-refh[0]).max()),float(np.abs(oh[1]-refh[1]).max()))*1e3
        dsl=max(float(np.abs(oh[2]-refh[2]).max()),float(np.abs(oh[3]-refh[3]).max()))
        return {"finite":finite,"max_pos_delta_um":dpos,"max_slope_delta_rad":dsl,"bit_exact":dpos==0.0 and dsl==0.0}
    def time_only(fn,n,blk):
        for _ in range(args.warmup): launch(fn,n,blk,out)
        dev.synchronize(); e0,e1=cp.cuda.Event(),cp.cuda.Event(); ts=[]
        for _ in range(args.repeats):
            e0.record(); launch(fn,n,blk,out); e1.record(); e1.synchronize(); ts.append(cp.cuda.get_elapsed_time(e0,e1))
        return ts

    results={}
    for key in names:
        fn=fns[key]; macs=MACS.get(key,DEF_MAC); acc=accuracy(fn)
        ko=pct(time_only(fn,N,args.block)); warp=pct(time_only(fn,32,32)); ns=ko["median"]*1e6/N
        a=attrs(fn,args.block)
        results[key]={"ns_per_track":ns,"single_warp_us":warp["median"]*1000,"macs_per_track":macs,
                      "pct_fp32_peak":100.0*(2*macs/(ns*1e-9))/peak,"resources":a,"accuracy_vs_ref":acc}
        print(f"{key:18s} {ns:7.3f} ns warp={warp['median']*1000:7.1f}us regs={a['num_regs']:3d} "
              f"spill={a['local_size_bytes']}B occ={a['occ']} %pk={results[key]['pct_fp32_peak']:5.1f} "
              f"dpos={acc['max_pos_delta_um']:.3g}um bitexact={acc['bit_exact']}")

    # block sweep for both fully-unrolled winners
    sweep={}
    for key in ("pinn_fused_fu","pinn_h64_fu"):
        sweep[key]={blk:pct(time_only(fns[key],N,blk))["median"]*1e6/N for blk in (64,96,128,192,256,384)}
        print(key,"sweep:",{k:round(v,3) for k,v in sweep[key].items()})

    base=results["pinn_ref"]["ns_per_track"]
    blob={"tier":1,"variant_set":"v3_fullunroll_ilp","n_tracks":int(N),"block_size":args.block,
          "warmup_iters":args.warmup,"timed_repeats":args.repeats,
          "published_baselines_ns_per_track":{"extrapUTT":2.344,"rk_field":5.709,"pinn_v2_baseline":7.054},
          "device_props":{"sm":sm,"clock_ghz":clk/1e6,"fp32_peak_tflop":peak/1e12},
          "methods":results,"speedup_vs_ref":{k:base/results[k]["ns_per_track"] for k in results},
          "block_sweep_ns":sweep}
    with open(args.out,"w") as f: json.dump(blob,f,indent=2)
    print("\nwrote",args.out)


if __name__ == "__main__":
    main()
