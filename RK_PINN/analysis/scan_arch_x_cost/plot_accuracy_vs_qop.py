#!/usr/bin/env python3
"""plot_accuracy_vs_qop.py — position accuracy vs SIGNED q/p for every scan model.

Median |Δpos| (µm) binned in the track's signed q/p (qop = X[:,4], the model's own
input variable). Shows the bend structure of the ~3 mm floor: small near q/p=0
(high p, ~straight) and rising toward large |q/p| (low p, strongly bent), roughly
symmetric in charge sign. Two panels: Block A (depth×activation), Block B (cost fn).

Outputs:  figures/fig_accuracy_vs_qop.png   results/accuracy_vs_qop.json
Reuses the tested loader from scan_eval_per_variable.py (run from this folder).
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scan_eval_per_variable import load_model, predict, BLOCK_A, BLOCK_B, LAB, TM, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")

with np.load(os.path.join(LAB, "data", "train_wave2_deploy.npz")) as d:
    X, Y, P = d["X"], d["Y"], d["P"]
ti = None
for _, run in BLOCK_A + BLOCK_B:
    f = os.path.join(TM, run, "test_indices.npy")
    if os.path.exists(f):
        ti = np.load(f); break
Xt, Yt, Pt = X[ti], Y[ti], P[ti]
qop = Xt[:, 4].astype(np.float64)
print(f"test split {len(ti)} tracks  device={DEV}  qop range [{qop.min():.3g}, {qop.max():.3g}]")

# p<->qop unit factor (p = k/|qop|), so we can label momentum on the q/p axis
k = float(np.median(np.abs(qop) * Pt))
edges = np.linspace(*np.percentile(qop, [0.5, 99.5]), 41)
cen = 0.5 * (edges[:-1] + edges[1:])


def median_vs_qop(dpos):
    idx = np.digitize(qop, edges) - 1
    out = np.full(len(cen), np.nan)
    for b in range(len(cen)):
        sel = idx == b
        if sel.sum() >= 50:
            out[b] = np.median(dpos[sel])
    return out


def eval_block(block):
    res = {}
    for lab, run in block:
        m = load_model(run)
        if m is None:
            print(f"  [skip] {run}"); continue
        pred = predict(m, Xt)
        dpos = np.hypot(pred[:, 0] - Yt[:, 0], pred[:, 1] - Yt[:, 1]) * 1e3  # µm
        res[run] = {"label": lab, "med_vs_qop": median_vs_qop(dpos).tolist()}
        print(f"  {run:18} done")
    return res


print("Block A:"); A = eval_block(BLOCK_A)
print("Block B:"); B = eval_block(BLOCK_B)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
for ax, blk, ttl in [(a1, A, "Block A — depth × activation"), (a2, B, "Block B — cost function")]:
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(blk), 1)))
    for (run, r), c in zip(blk.items(), cmap):
        ax.plot(cen, r["med_vs_qop"], "o-", color=c, ms=3, lw=1.5, label=r["label"])
    ax.set_yscale("log"); ax.set_xlabel("signed q/p  (qop = X[:,4])")
    ax.axhline(3000, color="k", ls="--", lw=.7); ax.text(cen[0], 3200, "3 mm floor", fontsize=8)
    ax.axhline(1000, color="g", ls=":", lw=.8); ax.text(cen[0], 1050, "1 mm target", fontsize=8, color="g")
    ax.axvline(0, color="gray", lw=.5)
    ax.set_title(ttl, fontsize=11); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, ncol=2)
    # momentum guide ticks on a twin top axis: p = k/|qop|
    axt = ax.secondary_xaxis("top")
    pticks = [2, 5, 20, 100]
    locs, labs = [], []
    for p in pticks:
        for s in (-1, 1):
            q = s * k / p
            if edges[0] <= q <= edges[-1]:
                locs.append(q); labs.append(f"{p:g}")
    axt.set_xticks(locs); axt.set_xticklabels(labs, fontsize=7)
    axt.set_xlabel("p [GeV] (=|c/qop|)", fontsize=8)
a1.set_ylabel("median |Δpos| [µm]")
fig.suptitle("Position accuracy vs signed q/p — the floor is the high-|q/p| (low-p) bend tail", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "fig_accuracy_vs_qop.png"); fig.savefig(out, dpi=130); plt.close(fig)
print("wrote", out)

json.dump({"n_test": int(len(ti)), "p_eq_k_over_absqop": k, "qop_bin_centers": cen.tolist(),
           "blockA": A, "blockB": B}, open(os.path.join(RES, "accuracy_vs_qop.json"), "w"), indent=2)
print("wrote results/accuracy_vs_qop.json")
