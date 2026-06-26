#!/usr/bin/env python3
"""gen_fused_kernels.py — emit a self-contained fused CUDA kernel per (shape, activation)
for the scan speed exam, matching the v3 throughput-winner pattern exactly:

  - NORM6 input normalisation + z_frac=1  (same as pinn_opt_kernels_v3.cu)
  - per-thread 1 track; layer0 fully unrolled; head folded into the last hidden loop
  - TWO variants per kernel: `_fused` (hidden j-loop rolled, inner unrolled, → LDC weight
    loads) and `_fu` (j-loop ALSO unrolled → weights become immediate FFMA operands).
    The last exam took pinn_fused (h96) and pinn_h64_fu (h64) → we report min over both.
  - kick_scaled_head envelope (g = exp(loggain) folded as literals; κ = 1e-3·qop·dz).

INFERENCE SPEED DEPENDS ONLY ON (shape, activation) — weight *values* don't change the
FFMA count or the transcendental cost — so placeholder weights give the true ns/track for
not-yet-trained shapes; real weights (when present) make each kernel a genuine deployable
artifact. Each kernel gets its OWN .cu (own __constant__ bank ≤64 KB).

Anchors: [96,96]/tanh and [64,64]/tanh use the real wave-2 weights and MUST reproduce the
published 4.85 ns / 0.91 ns — validating the generator before we trust new shapes/acts.
"""
import json, os, sys
import numpy as np

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/trained_models"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scan_kernels")
os.makedirs(OUT, exist_ok=True)

# distinct (shape, activation) inference points needed by the scan
SHAPES = []
for act in ("tanh", "silu", "gelu", "sin"):
    SHAPES.append((f"d2_{act}", [64, 64], act, "scanA_d2_" + act))
    SHAPES.append((f"d3_{act}", [64, 64, 64], act, "scanA_d3_" + act))
SHAPES.append(("h96_tanh", [96, 96], "tanh", "wave2_resid_h96"))   # Block-B / anchor

ACT_EXPR = {
    "tanh": "tanhf(a)",
    "silu": "(a/(1.f+expf(-a)))",
    "gelu": "(0.5f*a*(1.f+erff(a*0.7071067811865476f)))",
    "sin":  "sinf(30.f*a)",   # SIREN w0=30
}


def fetch_weights(run, dims):
    """Return (IM, IS, [W0,b0,W1,b1,...], Wc, bc, g[4]) from a trained run, or None."""
    import torch
    d = os.path.join(LAB, run)
    bm = os.path.join(d, "best_model.pt")
    if not os.path.exists(bm):
        return None
    sd = torch.load(bm, weights_only=False, map_location="cpu")["model_state_dict"]
    nz = os.path.join(d, "normalization.json")
    nrm = json.load(open(nz)) if os.path.exists(nz) else None
    IM = np.array(nrm["input_mean"][:5], np.float32) if nrm else np.zeros(5, np.float32)
    IS = np.array(nrm["input_std"][:5], np.float32) if nrm else np.ones(5, np.float32)
    lin = [2 * i for i in range(len(dims))]  # encoder.0,2,4 are the Linears
    W, B = [], []
    for li in lin:
        W.append(sd[f"encoder.{li}.weight"].numpy().astype(np.float32))
        B.append(sd[f"encoder.{li}.bias"].numpy().astype(np.float32))
    Wc = sd["correction_head.weight"].numpy().astype(np.float32)  # [4,H]
    bc = sd["correction_head.bias"].numpy().astype(np.float32)
    g = np.exp(sd["kick_loggain"].numpy().astype(np.float32)) if "kick_loggain" in sd else np.ones(4, np.float32)
    return IM, IS, W, B, Wc, bc, g


def placeholder(dims):
    rng = np.random.default_rng(0)
    W = [rng.standard_normal((dims[0], 6)).astype(np.float32) * 0.3]
    B = [np.zeros(dims[0], np.float32)]
    for k in range(1, len(dims)):
        W.append(rng.standard_normal((dims[k], dims[k-1])).astype(np.float32) * 0.1)
        B.append(np.zeros(dims[k], np.float32))
    Wc = (rng.standard_normal((4, dims[-1])).astype(np.float32) * 0.01)
    bc = np.zeros(4, np.float32)
    return (np.zeros(5, np.float32), np.ones(5, np.float32), W, B, Wc, bc, np.ones(4, np.float32))


def carr(name, a):
    a = np.asarray(a, np.float32).ravel()
    body = ",".join(f"{v:.8e}f" for v in a)
    return f"__constant__ float {name}[{a.size}]={{{body}}};"


def emit(label, dims, act, w):
    IM, IS, W, B, Wc, bc, g = w
    H = dims; nL = len(H); A = ACT_EXPR[act]
    consts = [carr("kIM", IM), carr("kIS", IS)]
    for k in range(nL):
        consts.append(carr(f"W{k}", W[k])); consts.append(carr(f"B{k}", B[k]))
    consts.append(carr("WC", Wc)); consts.append(carr("BC", bc))
    G = "".join(f"#define G{i} {g[i]:.8e}f\n" for i in range(4))

    # build the per-variant body
    def body(rolled):
        s = "  float h0[%d];\n" % H[0]
        # layer 0 (in6 -> h0), always inner+outer unrolled (matches v3)
        s += "  #pragma unroll\n  for(int o=0;o<%d;++o){ float a=B0[o];\n    #pragma unroll\n" % H[0]
        s += "    for(int i=0;i<6;++i) a=fmaf(W0[o*6+i],in6[i],a); h0[o]=%s; }\n" % A
        prev, prevN = "h0", H[0]
        # middle hidden layers (for d3): h0->h1, fully unrolled inner; outer rolled/unrolled
        for k in range(1, nL - 1):
            nm = "h%d" % k
            s += "  float %s[%d];\n" % (nm, H[k])
            s += ("  #pragma unroll\n" if not rolled else "")
            s += "  for(int o=0;o<%d;++o){ float a=B%d[o];\n    #pragma unroll\n" % (H[k], k)
            s += "    for(int i=0;i<%d;++i) a=fmaf(W%d[o*%d+i],%s[i],a); %s[o]=%s; }\n" % (
                prevN, k, prevN, prev, nm, A)
            prev, prevN = nm, H[k]
        # last hidden layer folds the head
        k = nL - 1
        s += "  float c0=BC[0],c1=BC[1],c2=BC[2],c3=BC[3];\n"
        s += ("  #pragma unroll\n" if not rolled else "")
        s += "  for(int j=0;j<%d;++j){ float a=B%d[j];\n    #pragma unroll\n" % (H[k], k)
        s += "    for(int i=0;i<%d;++i) a=fmaf(W%d[j*%d+i],%s[i],a);\n" % (prevN, k, prevN, prev)
        s += "    float hj=%s;\n" % A
        s += ("    c0=fmaf(WC[j],hj,c0); c1=fmaf(WC[%d+j],hj,c1); "
              "c2=fmaf(WC[%d+j],hj,c2); c3=fmaf(WC[%d+j],hj,c3); }\n" % (H[k], 2*H[k], 3*H[k]))
        return s

    pre = ("  TID;\n"
           "  const float x=X[t],y=Y[t],tx=TX[t],ty=TY[t],qop=QOP[t],dz=DZ[t];\n"
           "  const float in6[6]={(x-kIM[0])/kIS[0],(y-kIM[1])/kIS[1],(tx-kIM[2])/kIS[2],"
           "(ty-kIM[3])/kIS[3],(qop-kIM[4])/kIS[4],1.0f};\n")
    post = ("  const float kdz=1.0e-3f*qop*dz;\n"
            "  OTX[t]=tx+(G0*kdz)*c0; OTY[t]=ty+(G1*kdz)*c1;\n"
            "  OX[t]=x+tx*dz+(G2*kdz*dz)*c2; OY[t]=y+ty*dz+(G3*kdz*dz)*c3;\n")

    kfused = f"extern \"C\" __global__ void {label}_fused(SIG){{\n{pre}{body(True)}{post}}}\n"
    kfu    = f"extern \"C\" __global__ void {label}_fu(SIG){{\n{pre}{body(False)}{post}}}\n"

    HDR = ("#define TID const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=N) return;\n"
           "#define SIG const float* __restrict__ X,const float* __restrict__ Y,"
           "const float* __restrict__ TX,const float* __restrict__ TY,"
           "const float* __restrict__ QOP,const float* __restrict__ DZ,const int N,"
           "float* __restrict__ OX,float* __restrict__ OY,float* __restrict__ OTX,"
           "float* __restrict__ OTY\n")
    return HDR + G + "\n".join(consts) + "\n\n" + kfused + "\n" + kfu


def main():
    manifest = {}
    for label, dims, act, run in SHAPES:
        w = fetch_weights(run, dims)
        real = w is not None
        if not real:
            w = placeholder(dims)
        src = emit(label, dims, act, w)
        macs = 6 * dims[0] + sum(dims[k-1] * dims[k] for k in range(1, len(dims))) + dims[-1] * 4
        n_act = sum(dims)
        open(os.path.join(OUT, f"{label}.cu"), "w").write(src)
        manifest[label] = {"dims": dims, "act": act, "macs": macs, "n_act": n_act,
                           "weights": "real" if real else "placeholder", "run": run}
        print(f"{label:10s} {str(dims):14s} {act:5s} MACs={macs:6d} act/track={n_act:3d} "
              f"[{'real' if real else 'placeholder'}]")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("\nwrote", len(SHAPES), "kernels +", "manifest.json ->", OUT)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "models"))
    main()
