#!/usr/bin/env python3
"""plot_summary.py — cross-cut summary plots for the 2026-06-23 arch x cost scan.

Reads the 12 run history.json (LAB), the speed bench (scan_speed.json, L40S), and
results/scan_per_variable.json (written by scan_eval_per_variable.py — run that first).
Produces, in ./figures and ./results next to this file:
  figures/fig_accuracy_vs_speed.png      test UT->T median |dx| vs kernel speed
  figures/fig_accuracy_by_momentum.png   median |dpos| per momentum quartile, per run
  results/scan_summary_table.{json,csv}  one row per run (the headline numbers)

NB speeds are NVIDIA L40S (cc89); the extrapUTT / Nystrom / RK anchors were measured on
a V100 — different hardware, NOT directly comparable (see README).
"""
import json, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/trained_models"
SPEED = "/data/bfys/gscriven/Ex_rep/RK_PINN/results/scan_speed.json"

RUNS = ["scanA_d2_tanh", "scanA_d2_silu", "scanA_d2_gelu", "scanA_d2_sin",
        "scanA_d3_tanh", "scanA_d3_silu", "scanA_d3_gelu", "scanA_d3_sin",
        "scanB_alpha05", "scanB_delta2", "scanB_invp", "scanB_logcosh"]

spd = json.load(open(SPEED)); kern = spd["kernels"]; anchors = spd.get("anchors_published", {})


def speed_for(run):
    key = run.replace("scanA_", "").replace("scanB_", "")
    if key in kern:
        k = kern[key]; return k["best_ns_per_track"], k.get("weights", "?")
    # scanB_* are all [96,96] tanh: inference speed == h96_tanh (loss only affects training)
    if run.startswith("scanB_") and "h96_tanh" in kern:
        return kern["h96_tanh"]["best_ns_per_track"], "arch-proxy(h96)"
    return None, None


def converged(utt, best_ep, n_ep):
    # plateau heuristic on the val selection-metric curve
    if n_ep < 21:
        return "early-stop"
    imp20 = (utt[-21] - utt[-1]) / utt[-21]
    return "yes" if imp20 < 0.06 else "creeping"


rows = []
for run in RUNS:
    d = os.path.join(LAB, run)
    cfg = json.load(open(os.path.join(d, "config.json")))
    H = json.load(open(os.path.join(d, "history.json")))
    val = H["val"]; utt = [v["utt_median_dx_um"] for v in val]
    tf = H.get("test_final", {})
    ns, wq = speed_for(run)
    rows.append(dict(
        run=run, dims="x".join(map(str, cfg["hidden_dims"])), act=cfg["activation"],
        loss=cfg["loss"], depth=len(cfg["hidden_dims"]), n_ep=len(val),
        best_ep=H.get("best_epoch"), conv=converged(utt, H.get("best_epoch"), len(val)),
        test_utt_um=tf.get("utt_median_dx_um"), test_p95_um=tf.get("utt_p95_dx_um"),
        min_utt_um=min(utt), final_utt_um=utt[-1],
        speed_ns=ns, speed_wts=wq, train_min=round(H.get("training_time_s", 0) / 60, 1)))

# ---- table out ----
json.dump(rows, open(os.path.join(RES, "scan_summary_table.json"), "w"), indent=2)
with open(os.path.join(RES, "scan_summary_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

ACT_MK = {"tanh": "o", "silu": "s", "gelu": "^", "sin": "D"}
DEPTH_C = {2: "tab:blue", 3: "tab:red"}

# ---- plot 1: accuracy vs speed ----
fig, ax = plt.subplots(figsize=(9, 6.5))
for r in rows:
    if r["speed_ns"] is None or r["test_utt_um"] is None:
        continue
    real = (r["speed_wts"] == "real")
    ax.scatter(r["speed_ns"], r["test_utt_um"], marker=ACT_MK.get(r["act"], "o"),
               s=120, facecolors=(DEPTH_C.get(r["depth"], "gray") if real else "none"),
               edgecolors=DEPTH_C.get(r["depth"], "gray"), linewidths=1.8, zorder=3)
    ax.annotate(r["run"].replace("scan", ""), (r["speed_ns"], r["test_utt_um"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.axhline(3000, color="k", ls="--", lw=.8); ax.text(ax.get_xlim()[0], 3100, "3 mm floor", fontsize=8)
ax.axhline(1000, color="g", ls=":", lw=.9); ax.text(ax.get_xlim()[0], 1040, "1 mm target", fontsize=8, color="g")
ax.axhline(15, color="purple", ls="-.", lw=.9); ax.text(ax.get_xlim()[0], 16, "extrapUTT ~15 µm", fontsize=8, color="purple")
ax.set_yscale("log"); ax.set_xlabel("kernel speed [ns / track]  (L40S, best block/variant)")
ax.set_ylabel("test UT→T median |Δx| [µm]")
ax.set_title("Accuracy vs speed per architecture (filled=real weights, open=placeholder/arch-proxy)\n"
             "speeds L40S; extrapUTT/RK anchors were V100 — not directly comparable", fontsize=10)
# legend proxies
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker=m, color="gray", ls="", ms=9, label=a) for a, m in ACT_MK.items()]
leg += [Line2D([0], [0], marker="o", color=DEPTH_C[2], ls="", ms=9, label="depth 2"),
        Line2D([0], [0], marker="o", color=DEPTH_C[3], ls="", ms=9, label="depth 3")]
ax.legend(handles=leg, fontsize=8, ncol=2, loc="center right")
ax.grid(alpha=.3, which="both"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_accuracy_vs_speed.png"), dpi=130); plt.close(fig)
print("wrote figures/fig_accuracy_vs_speed.png")

# ---- plot 2: accuracy by momentum quartile ----
pv_path = os.path.join(RES, "scan_per_variable.json")
if os.path.exists(pv_path):
    pv = json.load(open(pv_path)); edges = pv["p_quartiles"]
    qlab = [f"Q1\n<{edges[0]:.1f}", f"Q2\n{edges[0]:.0f}-{edges[1]:.0f}",
            f"Q3\n{edges[1]:.0f}-{edges[2]:.0f}", f"Q4\n>{edges[2]:.0f}\nGeV"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(4)
    for ax, blk, ttl in [(a1, "blockA", "Block A — depth × activation"),
                         (a2, "blockB", "Block B — cost function")]:
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(pv[blk]), 1)))
        for (run, r), c in zip(pv[blk].items(), cmap):
            ax.plot(x, r["by_p_quartile_um"], "o-", color=c, lw=1.6, ms=5, label=r["label"])
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(qlab, fontsize=8)
        ax.axhline(3000, color="k", ls="--", lw=.7); ax.axhline(1000, color="g", ls=":", lw=.7)
        ax.set_xlabel("momentum quartile"); ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, ncol=2)
    a1.set_ylabel("median |Δpos| [µm]")
    fig.suptitle("Accuracy by momentum quartile — the ~3 mm floor is a low-p problem", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "fig_accuracy_by_momentum.png"), dpi=130); plt.close(fig)
    print("wrote figures/fig_accuracy_by_momentum.png")
else:
    print("[skip] fig_accuracy_by_momentum — run scan_eval_per_variable.py first")

print("wrote results/scan_summary_table.{json,csv}")
