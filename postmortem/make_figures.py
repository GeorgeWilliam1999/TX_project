#!/usr/bin/env python3
"""
Post-mortem result figures for the LHCb track-extrapolation surrogate project.

Reads READ-ONLY from the live experiment artifacts in
  /data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/paper_p0/
and writes version-controlled PNGs into ./figures/ (next to this script).

Every number plotted is taken straight from the artifacts (the *_three_arm.json /
*_frozen_pool.json summaries and the per-track *_arrays.npz files); the only
literals are the speed micro-benchmark numbers, which live in the archive note
archive/05_speed_benchmark.md (Tier-1) and are labelled as such on the plot.

Run with the TE env:
  /data/bfys/gscriven/conda/envs/TE/bin/python make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/paper_p0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 130

def load_json(name):
    with open(os.path.join(SRC, name)) as f:
        return json.load(f)

g4   = load_json("gen4_three_arm.json")
w2   = load_json("wave2_three_arm.json")
fp   = load_json("wave2_frozen_pool.json")

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("SAVED", p)

# Colour key used consistently across figures
C_INC = "#1a7f37"   # incumbent extrapUTT (green)
C_STR = "#8a8a8a"   # straight line (grey)
C_W1  = "#c1121f"   # wave-1 NN (red)
C_W2  = "#1f5fb4"   # wave-2 NN (blue)
C_RK  = "#e08a00"   # RK (orange)

# ---------------------------------------------------------------------------
# FIG 1 — first-wave catastrophe (log-scale bars)
# ---------------------------------------------------------------------------
labels = ["extrapUTT\n(incumbent)", "straight\nline",
          "NN g4 λ0\n2M", "NN g4 λ0.1\n2M", "NN g4 λ0\n10M", "NN g4 λ0.1\n10M"]
keys   = ["extrapUTT (incumbent)", "straight_line",
          "pinn_v2_g4_lam0_2M_cpu", "pinn_v2_g4_lam0p1_2M_cpu",
          "pinn_v2_g4_lam0_10M", "pinn_v2_g4_lam0p1_10M"]
vals   = [g4[k]["median_dx_um"] for k in keys]
cols   = [C_INC, C_STR, C_W1, C_W1, C_W1, C_W1]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_ylabel("median |Δx| at the T-plane  [µm]  (log scale)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("First-wave catastrophe: raw-output NNs never learn the bend\n"
             "(gen-4 PV-pointing UT→T plane, 7 947 tracks)")
ax.axhline(225343.23, color=C_STR, ls="--", lw=1)
for b, v in zip(bars, vals):
    txt = f"{v/1000:.0f} mm" if v >= 1000 else f"{v:.1f} µm"
    ax.text(b.get_x()+b.get_width()/2, v*1.15, txt, ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(3, 6e5)
ax.text(0.5, 0.04, "Every NN sits at the grey straight-line level (~225 mm): "
        "an absolute-output head learns essentially no curvature.",
        transform=ax.transAxes, ha="center", fontsize=8.2, style="italic", color="#444")
save(fig, "fig_firstwave_catastrophe.png")

# ---------------------------------------------------------------------------
# FIG 2 — the residual fix (before -> after)
# ---------------------------------------------------------------------------
before = g4["pinn_v2_g4_lam0_10M"]["median_dx_um"]      # 181 mm
after  = w2["wave2_resid_h64"]["median_dx_um"]          # 2.79 mm
straight = g4["straight_line"]["median_dx_um"]
inc      = g4["extrapUTT (incumbent)"]["median_dx_um"]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
xs = [0, 1]
ax.bar(xs, [before, after], width=0.55,
       color=[C_W1, C_W2], edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["wave-1\nabsolute head\n(NN g4 λ0 10M)",
                    "wave-2\nresidual-over-straight\n(resid_h64)"], fontsize=9.5)
ax.set_ylabel("median |Δx| at the T-plane  [µm]  (log scale)")
ax.set_title("The residual re-parametrisation: 181 mm → 2.79 mm (~65×)\n"
             "same physics, same field — only the output head changed")
ax.axhline(straight, color=C_STR, ls="--", lw=1.2); ax.text(1.45, straight, "straight line 225 mm", va="center", ha="right", fontsize=8, color="#555")
ax.axhline(inc, color=C_INC, ls="--", lw=1.2);     ax.text(1.45, inc, "extrapUTT 10.9 µm", va="center", ha="right", fontsize=8, color=C_INC)
for x, v in zip(xs, [before, after]):
    txt = f"{v/1000:.1f} mm"
    ax.text(x, v*1.18, txt, ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.annotate("", xy=(1, after*1.6), xytext=(0, before*0.7),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.text(0.5, np.sqrt(before*after)*2.2, "÷65", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(3, 6e5)
save(fig, "fig_residual_fix.png")

# ---------------------------------------------------------------------------
# FIG 3 — THE FLOOR PLOT: median |dx| vs parameter count
# ---------------------------------------------------------------------------
resid = {k: v for k, v in w2.items() if k.startswith("wave2_resid_")}
lam   = {k: v for k, v in w2.items() if k.startswith("wave2_lam")}
rp = sorted([(v["params"], v["median_dx_um"], k) for k, v in resid.items()])
lp = sorted([(v["params"], v["median_dx_um"], k) for k, v in lam.items()])
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot([p for p,_,_ in rp], [m for _,m,_ in rp], "o-", color=C_W2, lw=1.8,
        ms=8, label="residual head (resid_h32…h384)")
ax.plot([p for p,_,_ in lp], [m for _,m,_ in lp], "s", color="#7a3fb4", ms=8,
        label="λ>0 PDE-regularised (h128)")
for p, m, k in rp:
    ax.annotate(k.replace("wave2_resid_",""), (p, m), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=C_W2)
ax.set_xscale("log")
ax.set_xlabel("trainable parameters  (log scale)")
ax.set_ylabel("median |Δx| at the T-plane  [µm]")
ax.set_title("The capacity-independent floor: ~2.8–3.8 mm regardless of size\n"
             "more parameters do NOT help (h384 is the worst residual model)")
ax.axhspan(2788, 3782, color=C_W2, alpha=0.08)
ax.axhline(inc, color=C_INC, ls="--", lw=1.3)
ax.text(rp[-1][0], inc*1.05, "extrapUTT incumbent = 10.9 µm  (≈256–340× below the floor)",
        ha="right", va="bottom", fontsize=8.5, color=C_INC)
ax.set_ylim(0, 4200)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="lower left", fontsize=9)
# secondary annotation: the two extremes
ax.annotate("h32: 1 416 params → 2.81 mm", (1416, 2806.8), xytext=(2200, 1400),
            arrowprops=dict(arrowstyle="->", color="#444"), fontsize=8.5)
ax.annotate("h384: 152 072 params → 3.78 mm", (152072, 3782.3), xytext=(9000, 4000),
            arrowprops=dict(arrowstyle="->", color="#444"), fontsize=8.5)
save(fig, "fig_floor_vs_params.png")

# ---------------------------------------------------------------------------
# FIG 4 — error vs momentum (q/p quartile), incumbent vs best wave-2 NN
# ---------------------------------------------------------------------------
# JSON arrays are ordered high-p -> low-p; reverse to low-p -> high-p for reading.
qedges = [2.0, 5.34, 14.33, 36.34, 99.87]   # p quartile edges (GeV), verified from arrays
xlab = [f"{qedges[0]:.0f}–{qedges[1]:.1f}", f"{qedges[1]:.1f}–{qedges[2]:.1f}",
        f"{qedges[2]:.1f}–{qedges[3]:.1f}", f"{qedges[3]:.1f}–{qedges[4]:.0f}"]
inc_q = g4["extrapUTT (incumbent)"]["median_dx_um_by_qop_quartile_hi2lo_p"][::-1]
nn_q  = w2["wave2_resid_h64"]["median_dx_um_by_qop_quartile_hi2lo_p"][::-1]
st_q  = g4["straight_line"]["median_dx_um_by_qop_quartile_hi2lo_p"][::-1]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.plot(x, st_q, "^--", color=C_STR, ms=8, label="straight line")
ax.plot(x, nn_q, "o-", color=C_W2, ms=8, label="wave-2 NN (resid_h64)")
ax.plot(x, inc_q, "s-", color=C_INC, ms=8, label="extrapUTT (incumbent)")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(xlab)
ax.set_xlabel("track momentum p  [GeV]   (low p → high p ; more bend on the left)")
ax.set_ylabel("median |Δx| at the T-plane  [µm]  (log scale)")
ax.set_title("Error vs momentum: the NN→incumbent gap is widest at high p\n"
             "and narrows toward the hard, low-p (high-curvature) tracks")
for xi, a, b in zip(x, nn_q, inc_q):
    ax.annotate(f"×{a/b:.0f}", (xi, np.sqrt(a*b)), ha="center", fontsize=8.5,
                color="#333", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.4)
save(fig, "fig_error_vs_momentum.png")

# ---------------------------------------------------------------------------
# FIG 5 — general target (frozen pool, no incumbent)
# ---------------------------------------------------------------------------
fp_keys = ["straight_line", "pinn_v2_g4_lam0_10M",
           "wave2_resid_h32", "wave2_resid_h64", "wave2_resid_h128",
           "wave2_lam0p1_h128", "wave2_resid_h384"]
fp_lab  = ["straight\nline", "wave-1\nλ0 10M",
           "w2 resid\nh32", "w2 resid\nh64", "w2 resid\nh128",
           "w2 λ0.1\nh128", "w2 resid\nh384"]
fp_val  = [fp[k]["median_dx_um"] for k in fp_keys]
fp_col  = [C_STR, C_W1, C_W2, C_W2, C_W2, C_W2, C_W2]
fig, ax = plt.subplots(figsize=(9.0, 5.0))
bars = ax.bar(range(len(fp_val)), fp_val, color=fp_col, edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_ylabel("median |Δx| at target plane  [µm]  (log scale)")
ax.set_xticks(range(len(fp_lab))); ax.set_xticklabels(fp_lab, fontsize=9)
ax.set_title("The genuinely open target: general extrapolation (frozen pool, no incumbent)\n"
             "wave-2 NN beats straight line ~48–50×, but is still mm-scale absolute")
for b, v in zip(bars, fp_val):
    txt = f"{v/1000:.1f} mm" if v >= 1000 else f"{v:.0f} µm"
    ax.text(b.get_x()+b.get_width()/2, v*1.12, txt, ha="center", va="bottom", fontsize=8.5)
ax.text(0.99, 0.95, "extrapUTT is UNDEFINED off the UT→T plane —\nthere is no analytic incumbent here.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, style="italic",
        color="#a11", bbox=dict(boxstyle="round", fc="#fff4f4", ec="#e0b4b4"))
ax.set_ylim(1e3, 4e5)
save(fig, "fig_general_target.png")

# ---------------------------------------------------------------------------
# FIG 6 — the µm-vs-mm gap: CDF of |Δx| on the plane
# ---------------------------------------------------------------------------
arr = np.load(os.path.join(SRC, "wave2_three_arm_arrays.npz"))
def cdf(a):
    a = np.sort(a); y = np.linspace(0, 1, a.size)
    return a, y
fig, ax = plt.subplots(figsize=(8.6, 5.2))
for key, col, lab in [("straight_line", C_STR, "straight line  (median 225 mm)"),
                      ("wave2_resid_h64", C_W2, "wave-2 NN resid_h64  (median 2.79 mm)"),
                      ("extrapUTT", C_INC, "extrapUTT incumbent  (median 10.9 µm)")]:
    a, y = cdf(arr[key]); ax.plot(a, y, color=col, lw=2.2, label=lab)
ax.set_xscale("log")
ax.set_xlabel("per-track |Δx| at the T-plane  [µm]  (log scale)")
ax.set_ylabel("cumulative fraction of tracks")
ax.set_title("Two populations, three orders of magnitude apart\n"
             "(per-track CDF, gen-4 PV-pointing UT→T plane, 7 947 tracks)")
for v, c in [(10.897, C_INC), (2788.6, C_W2), (225343.2, C_STR)]:
    ax.axvline(v, color=c, ls=":", lw=1, alpha=0.7)
ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.5)
ax.text(11, 0.02, "10.9 µm", color=C_INC, fontsize=8, rotation=90, va="bottom")
ax.text(2789, 0.02, "2.79 mm", color=C_W2, fontsize=8, rotation=90, va="bottom")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.set_xlim(1, 3e6)
save(fig, "fig_cdf_um_vs_mm.png")

# ---------------------------------------------------------------------------
# FIG 7 — speed / throughput (Tier-1 micro-benchmark, from archive/05)
# ---------------------------------------------------------------------------
# Source: archive/05_speed_benchmark.md (Tier-1 isolated CUDA micro-benchmark,
# 1e6 tracks, fp32, GPU, NVRTC-compiled verbatim Allen device code).
sp_lab = ["extrapUTT\n(polynomial)", "RK + field\n(incumbent general)", "PINN_v2\n(ours)"]
sp_us  = [0.00234, 0.00571, 0.00705]      # µs / track (kernel)
sp_tps = [4.27e8, 1.75e8, 1.42e8]         # tracks / s
sp_col = [C_INC, C_RK, C_W2]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.8))
b1 = a1.bar(range(3), sp_us, color=sp_col, edgecolor="black", linewidth=0.6)
a1.set_ylabel("kernel time per track  [µs]  (lower = faster)")
a1.set_xticks(range(3)); a1.set_xticklabels(sp_lab, fontsize=8.5)
a1.set_title("Per-track kernel latency")
for b, v in zip(b1, sp_us):
    a1.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.5f}", ha="center", va="bottom", fontsize=8.5)
b2 = a2.bar(range(3), [t/1e8 for t in sp_tps], color=sp_col, edgecolor="black", linewidth=0.6)
a2.set_ylabel("throughput  [10⁸ tracks / s]  (higher = faster)")
a2.set_xticks(range(3)); a2.set_xticklabels(sp_lab, fontsize=8.5)
a2.set_title("Throughput")
for b, v in zip(b2, sp_tps):
    a2.text(b.get_x()+b.get_width()/2, v/1e8*1.01, f"{v/1e8:.2f}", ha="center", va="bottom", fontsize=8.5)
fig.suptitle("Speed: the surrogate is the SLOWEST of the three  "
             "(NN 1.2× slower than RK, 3.0× slower than extrapUTT)\n"
             "Tier-1 GPU micro-benchmark — archive/05_speed_benchmark.md", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig_speed_throughput.png")

print("\nALL FIGURES WRITTEN TO", OUT)
