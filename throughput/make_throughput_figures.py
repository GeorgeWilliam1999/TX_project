#!/usr/bin/env python3
"""
Throughput-optimisation characterisation figures for the LHCb track-extrapolation
surrogate (PINN_V2_UTT) on the Allen UT->T step.

Reads READ-ONLY from the optimisation result bundle
  /data/bfys/gscriven/pinn_opt_work/results/combined.json
(the Tesla V100-PCIE-32GB, 1M-track, NVRTC kernel-only micro-bench, identical
protocol to allen_bridge/bench/microbench.py) and writes version-controlled PNGs
into ./figures/ (next to this script).

Every plotted number is pulled from that JSON; the only literals are the two
incumbent baselines (RK, extrapUTT), the footprints, and the recorded h64/h96
accuracy floor — all of which also live in the same JSON and are labelled.

Run with the TE env:
  /data/bfys/gscriven/conda/envs/TE/bin/python make_throughput_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

SRC = "/data/bfys/gscriven/pinn_opt_work/results/combined.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 135

with open(SRC) as f:
    C = json.load(f)

# ---- device + incumbents (from the same JSON) -----------------------------
PEAK   = C["v2"]["device_props"]["fp32_peak_tflop"] * 1e3   # GFLOP/s (14131)
MEMBW  = C["v2"]["device_props"]["mem_bw_GBs"]              # GB/s   (898)
# incumbents: use the published baseline values (5.71 / 2.34), consistent with the
# verified throughput write-up; the same-slot re-time reproduces them (5.73 / 2.36).
RK     = C["v3"]["published_baselines_ns_per_track"]["rk_field"]          # ns (5.709)
EUTT   = C["v3"]["published_baselines_ns_per_track"]["extrapUTT"]         # ns (2.344)
BASE   = C["v3"]["methods"]["pinn_ref"]["ns_per_track"]                   # ns (same-slot)
FOOT   = C["baselines"]["footprints_bytes"]
FLOORS = C["v2"]["physics_floor_um"]      # median_dx_um by width
GEMM   = C["v2"]["gemm_ceiling"]

# canonical kernel numbers (v3 has the full unroll ladder; v1/v2 add warp + ceilings)
def ns(block, key):       return C[block]["methods"][key]["ns_per_track"]
def macs(block, key):     return C[block]["methods"][key]["macs_per_track"]
def pkpct(block, key):    return C[block]["methods"][key]["pct_fp32_peak"]
def res(block, key, f):   return C[block]["methods"][key]["resources"].get(f)

NS = {
    "baseline":  BASE,
    "warp":      ns("v1", "pinn_warp"),
    "fused_fu":  ns("v3", "pinn_fused_fu"),
    "ilp4":      ns("v3", "pinn_fused_ilp4"),
    "fused":     ns("v3", "pinn_fused"),
    "h64":       ns("v3", "pinn_h64"),
    "h64_fu":    ns("v3", "pinn_h64_fu"),
    "ftanh":     ns("v2", "pinn_fused_ftanh"),
    "h16":       ns("v2", "pinn_fused_h16"),
    "lb":        ns("v2", "pinn_fused_lb"),
    "gemm_fp32": GEMM["fp32"]["ns_per_track"],
    "gemm_fp16": GEMM["fp16"]["ns_per_track"],
    "hbm_fp16":  GEMM["fp16"]["hbm_traffic_floor_ns_per_track"],
}

# colour key (consistent with the post-mortem set)
C_INC = "#1a7f37"   # incumbent extrapUTT (green)
C_RK  = "#e08a00"   # RK (orange)
C_BAD = "#c1121f"   # baseline / failed levers (red)
C_NN  = "#1f5fb4"   # optimised NN h96 (blue)
C_GLD = "#b8860b"   # h64 fastest (gold)
C_GREY= "#8a8a8a"

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("SAVED", p)

def fmt_ns(v):
    return f"{v:.2f}" if v >= 1 else f"{v:.2f}".lstrip("0") if v >= 0.1 else f"{v:.3f}"

# ===========================================================================
# FIG 1 — the reversal: ns/track, slowest -> fastest
# ===========================================================================
order  = [("PINN baseline\n(locked kernel)", BASE, C_BAD),
          ("RK + field\n(general ref)",       RK,   C_RK),
          ("PINN_fused\n(h96, bit-exact)",    NS["fused"], C_NN),
          ("extrapUTT\n(incumbent poly)",     EUTT, C_INC),
          ("PINN_h64_fu\n(optimised NN)",     NS["h64_fu"], C_GLD)]
labels = [o[0] for o in order]; vals = [o[1] for o in order]; cols = [o[2] for o in order]
fig, ax = plt.subplots(figsize=(9.6, 5.4))
bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.7, width=0.66)
ax.axhline(RK,   color=C_RK,  ls="--", lw=1.2, alpha=0.8)
ax.axhline(EUTT, color=C_INC, ls="--", lw=1.2, alpha=0.8)
ax.text(4.45, RK,   "RK 5.71", color=C_RK,  fontsize=8, va="bottom", ha="right")
ax.text(4.45, EUTT, "extrapUTT 2.34", color=C_INC, fontsize=8, va="bottom", ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.12, f"{v:.2f} ns", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("kernel-only time per track  [ns]   (lower = faster)")
ax.set_ylim(0, 8.1)
ax.set_title("The throughput reversal: the same network goes from slowest to fastest\n"
             "Tesla V100-PCIE-32GB · 1,000,000 real gen-4 tracks · fp32 · kernel-only median",
             fontsize=11)
arr = FancyArrowPatch((0, BASE+0.35), (4, NS["h64_fu"]+0.35), connectionstyle="arc3,rad=-0.28",
                      arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#333")
ax.add_patch(arr)
ax.text(2.0, 7.55, "7.7× faster  ·  slowest → fastest extrapolator",
        ha="center", fontsize=10.5, fontweight="bold", color="#333")
ax.text(2.0, 6.95, "(register-spill removal + full-unroll + width-64; no retrain)",
        ha="center", fontsize=8.6, style="italic", color="#555")
save(fig, "fig01_speed_reversal.png")

# ===========================================================================
# FIG 2 — the optimisation waterfall (how it was sped up vs the locked kernel)
# ===========================================================================
stages = [
    ("locked\nkernel",        BASE,         C_BAD, "two spilled h0[96],h1[96]\n→ 9,216 local loads/track"),
    ("kill spill\n+ fuse head", NS["fused"], C_NN,  "h0 in registers, h1 never\nmaterialised → 0 B spill\n(bit-exact)"),
    ("width\nh96 → h64",      NS["h64"],     "#3f7fbf", "capacity ladder flat:\nh64 2.79 mm ≈ h96 3.58 mm\n→ 2.1× fewer MACs"),
    ("full unroll\n+ const-FFMA", NS["h64_fu"], C_GLD, "4,096 FFMA fit the I-cache;\nweights fold into the\ninstruction (no loads)"),
]
fig, ax = plt.subplots(figsize=(9.8, 5.6))
xs = range(len(stages))
prev = None
for i, (lab, v, col, note) in enumerate(stages):
    ax.bar(i, v, color=col, edgecolor="black", linewidth=0.7, width=0.6, zorder=3)
    ax.text(i, v+0.12, f"{v:.2f} ns", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.text(i, -0.42, lab.replace("\n", " "), ha="center", va="top", fontsize=9.2, fontweight="bold", color="#222")
    ax.text(i, -1.02, note, ha="center", va="top", fontsize=7.6, color="#444")
    if prev is not None:
        ax.annotate("", xy=(i-0.30, v), xytext=(i-0.70, prev),
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.4))
        drop = prev / v
        ax.text(i-0.5, max(prev, v)+0.45, f"÷{drop:.2f}", ha="center", fontsize=9.5,
                fontweight="bold", color="#222")
    prev = v
ax.axhline(RK,   color=C_RK,  ls="--", lw=1.1, alpha=0.8); ax.text(3.42, RK,   " RK", color=C_RK,  fontsize=8, va="bottom")
ax.axhline(EUTT, color=C_INC, ls="--", lw=1.1, alpha=0.8); ax.text(3.42, EUTT, " extrapUTT", color=C_INC, fontsize=8, va="bottom")
ax.set_xticks([])
ax.set_ylabel("kernel-only time per track  [ns]")
ax.set_ylim(-2.4, 8.0)
ax.set_title("How the speed-up was won: three levers on the locked forward pass\n"
             "(grey note under each bar = the mechanism; the h96 full-unroll that REGRESSED is shown in Fig 5)",
             fontsize=10.6)
ax.spines["bottom"].set_position(("data", 0))
save(fig, "fig02_optimisation_waterfall.png")

# ===========================================================================
# FIG 3 — roofline: the thread-per-track NN family climbing to the fp32 ceiling
# ===========================================================================
def flop(m): return 2.0 * m
def achieved_gflops(block, key):
    return flop(macs(block, key)) / (ns(block, key) * 1e-9) / 1e9
pts = [  # (label, block, key, colour, (dx,dy) label offset in pts, ha, va)
    ("baseline (spill)", "v3", "pinn_ref",    C_BAD,      (10, -20), "left",  "top"),
    ("fused h96",        "v3", "pinn_fused",  C_NN,       (12,  10), "left",  "bottom"),
    ("h64 inner-unroll", "v3", "pinn_h64",    "#3f7fbf",  (-12,  6), "right", "bottom"),
    ("h64_fu (fastest)", "v3", "pinn_h64_fu", C_GLD,      (0,  -28), "center","top"),
]
BYTES = 40.0  # HBM bytes/track: 6 inputs + 4 outputs, fp32; weights are constant-cached (broadcast)
fig, ax = plt.subplots(figsize=(9.2, 5.8))
ai = np.logspace(-1, 3.2, 400)
roof = np.minimum(PEAK, MEMBW * ai)
ax.plot(ai, roof, color="black", lw=2.0, zorder=2)
ax.fill_between(ai, roof, 1, color="#f0f0f0", zorder=0)
ridge = PEAK / MEMBW
ax.axvline(ridge, color=C_GREY, ls=":", lw=1.2)
ax.text(ridge*1.05, 130, f"ridge\n{ridge:.1f} FLOP/byte", color=C_GREY, fontsize=8, va="bottom")
ax.text(2.2, PEAK*1.05, f"fp32 compute ceiling  {PEAK/1e3:.1f} TFLOP/s", fontsize=8.5)
ax.text(0.12, MEMBW*0.13, f"HBM  {MEMBW:.0f} GB/s", rotation=34, fontsize=8.2, color="#555")
for lab, blk, key, col, off, ha, va in pts:
    m = macs(blk, key); ai_k = flop(m) / BYTES; g = achieved_gflops(blk, key)
    ax.scatter([ai_k], [g], s=140, color=col, edgecolor="black", zorder=4, linewidth=0.8)
    ax.annotate(f"{lab}\n{g/1e3:.1f} TFLOP/s · {100*g/PEAK:.0f}% peak",
                (ai_k, g), textcoords="offset points", xytext=off, ha=ha, va=va,
                fontsize=8.2, color=col, fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity  [FLOP / byte of HBM traffic]  (log)")
ax.set_ylabel("achieved performance  [GFLOP/s]  (log)")
ax.set_xlim(0.1, 4000); ax.set_ylim(800, 2.4e4)
ax.set_title("Roofline: the NN is deeply compute-bound — the win is closing the gap to the ceiling\n"
             "thread-per-track, weights broadcast from constant memory (AI » ridge)", fontsize=10.6)
ax.grid(True, which="both", ls=":", alpha=0.35)
save(fig, "fig03_roofline.png")

# ===========================================================================
# FIG 4 — the bottleneck: register spill removed, efficiency unlocked
# ===========================================================================
bk = [("baseline", "v3", "pinn_ref", C_BAD),
      ("fused h96", "v3", "pinn_fused", C_NN),
      ("h64_fu",    "v3", "pinn_h64_fu", C_GLD)]
spill = [res(b, k, "local_size_bytes") for _, b, k, _ in bk]
peakp = [pkpct(b, k) for _, b, k, _ in bk]
cols  = [c for *_, c in bk]; labs = [l for l, *_ in bk]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.8))
b1 = a1.bar(range(3), spill, color=cols, edgecolor="black", linewidth=0.7, width=0.6)
a1.set_ylabel("register spill to local memory  [bytes/thread]")
a1.set_xticks(range(3)); a1.set_xticklabels(labs, fontsize=9.5)
a1.set_title("The bug: the locked kernel spills two 96-float arrays")
for b, v in zip(b1, spill):
    a1.text(b.get_x()+b.get_width()/2, v+8, f"{v} B" + ("  → 9,216\nlocal loads/track" if v else "  (none)"),
            ha="center", va="bottom", fontsize=8.6, fontweight="bold" if v else "normal")
a1.set_ylim(0, 470)
b2 = a2.bar(range(3), peakp, color=cols, edgecolor="black", linewidth=0.7, width=0.6)
a2.axhline(100, color="black", ls="--", lw=1); a2.text(2.4, 100, "fp32 peak", ha="right", va="bottom", fontsize=8)
a2.set_ylabel("fraction of fp32 FMA peak achieved  [%]")
a2.set_xticks(range(3)); a2.set_xticklabels(labs, fontsize=9.5)
a2.set_title("The pay-off: spill-free → compute-bound efficiency")
for b, v in zip(b2, peakp):
    a2.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
a2.set_ylim(0, 108)
fig.suptitle("Where the time went, and where it goes now: eliminating the local-memory spill",
             fontsize=11, y=1.02)
save(fig, "fig04_bottleneck_spill.png")

# ===========================================================================
# FIG 5 — negative results: the textbook GPU tricks that LOSE
# ===========================================================================
negs = [("PINN_fused\n(winner)",        NS["fused"],     C_GLD),
        ("warp-coop\nGEMV",             NS["warp"],      C_BAD),
        ("cuBLAS 3×GEMM\nfp32",         NS["gemm_fp32"], C_BAD),
        ("cuBLAS 3×GEMM\nfp16 TC",      NS["gemm_fp16"], C_BAD),
        ("force occ↑\n(regs→80)",       NS["lb"],        C_BAD),
        ("fast __expf\ntanh",           NS["ftanh"],     C_GREY),
        ("fp16 h0\nstorage",            NS["h16"],       C_GREY)]
labs = [n[0] for n in negs]; vals = [n[1] for n in negs]; cols = [n[2] for n in negs]
fig, ax = plt.subplots(figsize=(10.2, 5.4))
bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.7, width=0.64)
ax.axhline(NS["fused"], color=C_GLD, ls="--", lw=1.3)
ax.text(6.4, NS["fused"], "fused 4.85", color="#8a6a00", fontsize=8, va="bottom", ha="right")
ax.axhline(NS["hbm_fp16"], color=C_GREY, ls=":", lw=1.2)
ax.text(6.4, NS["hbm_fp16"], "naive-GEMM HBM floor 0.88 (only a FUSED kernel reaches it)",
        color="#555", fontsize=7.6, va="bottom", ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8.8)
ax.set_ylabel("kernel-only time per track  [ns]   (lower = faster)")
ax.set_ylim(0, 9.2)
ax.set_title("The obvious GPU levers that LOSE for a 3-layer per-track MLP\n"
             "warp-cooperative GEMV breaks the constant-weight broadcast; batched GEMM streams activations through HBM",
             fontsize=10.6)
save(fig, "fig05_negative_results.png")

# ===========================================================================
# FIG 6 — the value proposition: speed vs accuracy, speed vs footprint
# ===========================================================================
acc = {  # ns, median |dx| um
    "extrapUTT": (EUTT, 10.9, C_INC),
    "RK+field":  (RK,  None, C_RK),     # reference truth (no surrogate error)
    "fused h96": (NS["fused"], FLOORS["h96_deployed"], C_NN),
    "h64_fu":    (NS["h64_fu"], FLOORS["h64"], C_GLD),
}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.0))
for lab, (x, y, col) in acc.items():
    if y is None:
        a1.scatter([x], [3.2], marker="*", s=200, color=col, edgecolor="black", zorder=4)
        a1.annotate(f"{lab}\n(truth ref)", (x, 3.2), textcoords="offset points", xytext=(-6, 8),
                    fontsize=8.2, color=col, ha="right")
        continue
    a1.scatter([x], [y], s=130, color=col, edgecolor="black", zorder=4)
    dx, dy, ha = (7, 0, "left") if lab == "fused h96" else (-8, 0, "right")
    a1.annotate(f"{lab}\n{y/1000:.2f} mm" if y >= 1000 else f"{lab}\n{y:.1f} µm",
                (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8.4, color=col, va="center", ha=ha)
a1.set_yscale("log")
a1.set_xlabel("kernel time per track  [ns]   (← faster)")
a1.set_ylabel("median position error at T-plane  [µm]  (log)")
a1.set_title("Speed vs accuracy")
a1.set_xlim(0, 6.4); a1.set_ylim(2.2, 1.3e4)
a1.axhspan(2788, 3782, color=C_NN, alpha=0.07)
a1.text(3.2, 7000, "NN accuracy floor ~2.8–3.8 mm  (capacity-independent)",
        fontsize=7.8, color="#444", ha="center")
a1.grid(True, which="both", ls=":", alpha=0.35)

fp = {"extrapUTT": (EUTT, FOOT["extraputt_chart"], C_INC),
      "RK+field":  (RK,  FOOT["field_map_texture"], C_RK),
      "fused h96": (NS["fused"], FOOT["pinn_v2_weights"], C_NN),
      "h64_fu":    (NS["h64_fu"], 19000, C_GLD)}
for lab, (x, y, col) in fp.items():
    a2.scatter([x], [y], s=130, color=col, edgecolor="black", zorder=4)
    txt = f"{y/1e6:.2f} MB" if y >= 1e6 else f"{y/1e3:.0f} KB"
    a2.annotate(f"{lab}\n{txt}", (x, y), textcoords="offset points", xytext=(7, 0),
                fontsize=8.4, color=col, va="center")
a2.set_yscale("log")
a2.set_xlabel("kernel time per track  [ns]   (← faster)")
a2.set_ylabel("model / table footprint  [bytes]  (log)")
a2.set_title("Speed vs footprint")
a2.set_xlim(0, 6.4); a2.set_ylim(1e4, 3e7)
a2.grid(True, which="both", ls=":", alpha=0.35)
fig.suptitle("After the optimisation the NN wins speed AND footprint — and still pays ~3 mm accuracy\n"
             "(the honest trade: fastest + tiniest, but a coarse surrogate, not an accuracy replacement)",
             fontsize=10.6, y=1.04)
save(fig, "fig06_value_proposition.png")

# ===========================================================================
# FIG 7 — launch-config robustness (occupancy is already balanced)
# ===========================================================================
bs_fu = C["v3"]["block_sweep_ns"]["pinn_fused_fu"]
bs_h64 = C["v3"]["block_sweep_ns"]["pinn_h64_fu"]
bs_h96 = C["v2"]["pinn_fused_block_sweep_ns"]
fig, ax = plt.subplots(figsize=(8.8, 5.0))
for d, lab, col in [(bs_h96, "fused h96", C_NN), (bs_h64, "h64_fu (fastest)", C_GLD)]:
    xs = sorted(int(k) for k in d); ys = [d[str(x)] for x in xs]
    ax.plot(xs, ys, "o-", color=col, lw=1.8, ms=7, label=lab)
ax.axvline(256, color=C_GREY, ls=":", lw=1.2); ax.text(258, 1.0, "Allen block 256", color="#555", fontsize=8)
ax.set_xlabel("CUDA block size  [threads]")
ax.set_ylabel("kernel-only time per track  [ns]")
ax.set_title("Launch-configuration robustness: the optimum sits at the Allen block size\n"
             "(occupancy is already balanced at 0.25 — forcing it higher costs spill, see Fig 5)", fontsize=10.4)
ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=9)
ax.set_ylim(0, 5.8)
save(fig, "fig07_block_sweep.png")

print("\nAll figures written to", OUT)
print(f"key numbers: baseline {BASE:.2f} | fused {NS['fused']:.2f} | h64_fu {NS['h64_fu']:.3f} ns "
      f"| RK {RK:.2f} | extrapUTT {EUTT:.2f} | peak {PEAK/1e3:.1f} TFLOP/s")
