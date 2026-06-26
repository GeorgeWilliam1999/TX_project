#!/usr/bin/env python3
"""plot_nn_vs_allen.py — figures for the NN-vs-Allen speed/accuracy head-to-head.

Reads results/comparison_nn_vs_allen.json + comparison_arrays.npz. Produces:
  fig_accuracy_bar.png        median |dpos| per method (log) vs the 3 mm floor
  fig_accuracy_by_p.png       median |dpos| vs momentum quartile (the low-p tail)
  fig_speed_vs_accuracy.png   the Pareto: ns/track vs median |dpos| (the money plot)
  fig_accuracy_cdf.png        CDF of |dpos| per method
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
blob = json.load(open(os.path.join(R, "comparison_nn_vs_allen.json")))
arr = np.load(os.path.join(R, "comparison_arrays.npz"))
res = blob["results"]

# method -> (display, array-key, GPU ns/track, family)  family: nn / allen / control
# NOTE: the locked candidate/pinn_v2_ALLEN_v1 is the WEAK-FIELD lock (240 mm on the real
# field) -> excluded as "our NN"; the accurate deployable is a wave-2 retrain (h64/h96).
# NN GPU speeds are the optimised-kernel numbers (architecture, weight-independent):
# h64_fu 0.91 ns, fused h96 4.85 ns.
import os as _os
NYS = float(_os.environ.get("NYSTROM_NS", "0")) or None
SPEED = {
    "NN h64 (accuracy-equiv)":            ("NN h64 (fastest)",          "NN h64 (accuracy-equiv)",     0.91, "nn"),
    "NN h96 (wave2)":                     ("NN h96 (fused)",            "NN h96 (wave2)",              4.85, "nn"),
    "extrapUTT (Allen fast UT->T)":       ("extrapUTT (Allen fast)",    "extrapUTT",                   2.34, "allen"),
    "Allen RK Cash-Karp (deployed, bug)": ("Allen RK (deployed)",       "RK_deployed",                 5.71, "allen"),
    "Allen RK Cash-Karp (corrected)":     ("Allen RK (corrected*)",     "RK_corrected",                5.71, "allen"),
    "Allen Nystrom fast-step":            ("Allen Nystrom (fast)",      "Nystrom",                     NYS,  "allen"),
    "straight line (control)":            ("straight line",             "straight_line",               0.05, "control"),
}
COL = {"nn": "tab:red", "allen": "tab:blue", "control": "gray"}
p = arr["p_GeV"]

# ---------- 1. accuracy bar ----------
items = [(disp, res[k]["median_pos_um"], fam) for k, (disp, ak, sp, fam) in SPEED.items() if k in res]
items.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([d for d, _, _ in items], [v for _, v, _ in items],
       color=[COL[f] for _, _, f in items])
for i, (_, v, _) in enumerate(items):
    ax.text(i, v * 1.1, f"{v:.1f}" if v < 1000 else f"{v/1000:.1f}mm", ha="center", fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("median |Δpos| vs truth  [µm]")
ax.axhline(3000, color="k", ls="--", lw=.8); ax.text(0, 3300, "3 mm surrogate floor", fontsize=8)
ax.set_title("UT→T accuracy: our NN vs Allen extrapolators (plane-ref, vs truth)")
plt.xticks(rotation=20, ha="right"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig_accuracy_bar.png"), dpi=130); plt.close(fig)

# ---------- 2. accuracy by momentum quartile ----------
fig, ax = plt.subplots(figsize=(8, 5))
xq = [1, 2, 3, 4]  # lo -> hi
for k, (disp, ak, sp, fam) in SPEED.items():
    if k not in res: continue
    ax.plot(xq, res[k]["median_pos_um_by_p_quartile_lo2hi"], "o-", color=COL[fam], label=disp,
            lw=2 if fam == "nn" else 1.3, ms=6 if fam == "nn" else 4)
ax.set_yscale("log"); ax.set_xticks(xq); ax.set_xticklabels(["Q1\nlow p", "Q2", "Q3", "Q4\nhigh p"])
ax.set_ylabel("median |Δpos| [µm]"); ax.set_title("Accuracy vs momentum — the low-p (Kalman-critical) tail")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, which="both"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig_accuracy_by_p.png"), dpi=130); plt.close(fig)

# ---------- 3. speed vs accuracy Pareto ----------
OFF = {"NN h64 (accuracy-equiv)": (8, 8), "NN h96 (wave2)": (-12, -22),
       "Allen RK Cash-Karp (deployed, bug)": (8, 8), "extrapUTT (Allen fast UT->T)": (8, 8),
       "Allen RK Cash-Karp (corrected)": (-10, 10), "Allen Nystrom fast-step": (8, 8),
       "straight line (control)": (8, 6)}
fig, ax = plt.subplots(figsize=(9.5, 6.5))
for k, (disp, ak, sp, fam) in SPEED.items():
    if k not in res or sp is None: continue
    acc = res[k]["median_pos_um"]
    ax.scatter([sp], [acc], s=150, color=COL[fam], edgecolor="k", zorder=5)
    txt = f"{disp}\n{sp:.2f} ns, {acc:.1f} µm" if acc < 1000 else f"{disp}\n{sp:.2f} ns, {acc/1000:.1f} mm"
    ax.annotate(txt, (sp, acc), textcoords="offset points", xytext=OFF.get(k, (8, 6)), fontsize=8,
                ha="right" if OFF.get(k, (8, 6))[0] < 0 else "left")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("GPU speed  [ns/track]  (V100, ←faster)")
ax.set_ylabel("median |Δpos| vs truth  [µm]  (↓more accurate)")
ax.set_title("Speed vs accuracy (V100, UT→T) — Allen's FAST Nyström dominates our NN on BOTH axes\n"
             "(0.41 ns & 0.5 mm  vs  0.91 ns & 3.6 mm); the NN wins only on footprint (40 KB vs 11.5 MB)")
ax.grid(alpha=.3, which="both"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig_speed_vs_accuracy.png"), dpi=130); plt.close(fig)

# ---------- 4. CDF of |dpos| ----------
fig, ax = plt.subplots(figsize=(8, 5))
for k, (disp, ak, sp, fam) in SPEED.items():
    if ak not in arr.files: continue
    a = np.sort(arr[ak]); cdf = np.linspace(0, 1, len(a))
    ax.plot(a, cdf, color=COL[fam], lw=2 if fam == "nn" else 1.3, label=disp)
ax.set_xscale("log"); ax.set_xlabel("|Δpos| vs truth [µm]"); ax.set_ylabel("fraction of tracks ≤")
ax.axvline(3000, color="k", ls="--", lw=.7); ax.set_title("Per-track error distribution (CDF)")
ax.legend(fontsize=7); ax.grid(alpha=.3, which="both"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig_accuracy_cdf.png"), dpi=130); plt.close(fig)

print("wrote 4 figures to figures/")
for k in res:
    print(f"  {k:<38} median={res[k]['median_pos_um']:.1f} um")
