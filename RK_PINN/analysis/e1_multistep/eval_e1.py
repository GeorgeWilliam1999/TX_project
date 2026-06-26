#!/usr/bin/env python3
"""Accuracy test harness for the bend-parametrisation study (UT->T extrapolation).

Compares, on the shared seed-42 held-out split (fp32, Allen-faithful):
  * the single-shot baseline (h96 kick head) and the best single-shot net,
  * the field-free MULTI-STEP UNROLL of the same kick head (loop it N times),
  * the unroll WITH the 2nd-order curvature term (higher-order kick basis).

Metric = UT->T-subset median |dx| (the headline, from history.json test_final) plus
the all-pairs median / p95 / by-momentum-quartile and median |Δpos| vs signed q/p.
Run dirs keep their original names (e1_unroll*/e2_order2_*) for provenance.

Outputs: figures/fig_e1_vs_qop.png, results/e1_eval.json
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"
TM = os.path.join(LAB, "trained_models")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "models"))
from architectures import create_model

# (plot label, trained_models dir, kind). dir names are kept for provenance; labels are plain.
# kind drives the plot style: single-shot (dashed) · multi-step unroll (solid) · unroll+curvature (dash-dot).
RUNS = [
    ("single-shot baseline (h96)",       "wave2_resid_h96",    "single"),
    ("best single-shot (depth-3 tanh)",  "scanA_d3_tanh",      "single"),
    ("multi-step unroll ×2",             "e1_unroll2",         "unroll"),
    ("multi-step unroll ×4",             "e1_unroll4",         "unroll"),
    ("multi-step unroll ×8",             "e1_unroll8",         "unroll"),
    ("multi-step unroll ×8 (long)",      "e1_unroll8_long",    "unroll"),
    ("multi-step unroll ×16",            "e1_unroll16",        "unroll"),
    ("multi-step unroll ×32",            "e1_unroll32",        "unroll"),
    ("unroll ×8 + 2nd-order curvature",  "e2_order2_unroll8",  "curv"),
    ("unroll ×16 + 2nd-order curvature", "e2_order2_unroll16", "curv"),
]
N_SUB = 200_000


def utt_metrics(run):
    """The UT->T-subset headline (utt_median_dx_um) recorded at train time, from history.json test_final."""
    h = json.load(open(os.path.join(TM, run, "history.json")))
    tf = h.get("test_final", {}) or {}
    return (tf.get("utt_median_dx_um"), tf.get("utt_p95_dx_um"),
            h.get("best_epoch"), len(h.get("val", [])))


def load(run):
    d = os.path.join(TM, run); c = json.load(open(os.path.join(d, "config.json")))
    m = create_model("pinn_v2", hidden_dims=c["hidden_dims"], activation=c["activation"],
                     dropout=c.get("dropout", 0.0), lambda_pde=c.get("lambda_pde", 0.0),
                     lambda_ic=c.get("lambda_ic", 0.0), n_collocation=c.get("n_collocation", 2),
                     kick_scaled_head=c.get("kick_scaled_head", False),
                     pde_scale_mode=c.get("pde_scale_mode", "legacy"),
                     pde_ref_length=c.get("pde_ref_length", 5161.0),
                     siren_w0=c.get("siren_w0", 30.0), n_unroll=c.get("n_unroll", 1),
                     kick_order=c.get("kick_order", 1))
    m.load_normalization(os.path.join(d, "normalization.json"))
    ck = torch.load(os.path.join(d, "best_model.pt"), weights_only=False, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, int(c.get("n_unroll", 1))


with np.load(os.path.join(LAB, "data", "train_wave2_deploy.npz")) as d:
    X, Y, P = d["X"], d["Y"], d["P"]
ti = np.load(os.path.join(TM, "scanA_d2_tanh", "test_indices.npy"))
rng = np.random.default_rng(7); ti = rng.choice(ti, size=min(N_SUB, len(ti)), replace=False)
Xt, Yt, Pt = X[ti], Y[ti], P[ti]
qop = Xt[:, 4].astype(np.float64)
k = float(np.median(np.abs(qop) * Pt))
pq = np.percentile(Pt, [25, 50, 75])
edges = np.linspace(*np.percentile(qop, [0.5, 99.5]), 41); cen = 0.5 * (edges[:-1] + edges[1:])
print(f"eval on {len(ti)} tracks (fp32, CPU)")


@torch.no_grad()
def predict(m):
    out = np.empty((len(ti), 4), np.float64)
    for i in range(0, len(ti), 100000):
        xb = torch.from_numpy(Xt[i:i+100000].astype(np.float32))
        out[i:i+100000] = m(xb).numpy()[:, :4]
    return out


def med_vs_qop(dpos):
    idx = np.digitize(qop, edges) - 1
    o = np.full(len(cen), np.nan)
    for b in range(len(cen)):
        sel = idx == b
        if sel.sum() >= 50: o[b] = np.median(dpos[sel])
    return o


res = {}; fig, ax = plt.subplots(figsize=(10.5, 6.5))
unroll_runs = [d for _, d, k in RUNS if k == "unroll"]
unroll_cmap = {d: c for d, c in zip(unroll_runs, plt.cm.plasma(np.linspace(0.05, 0.82, len(unroll_runs))))}
single_colors = {"wave2_resid_h96": "0.2", "scanA_d3_tanh": "0.55"}
curv_colors = {"e2_order2_unroll8": "tab:cyan", "e2_order2_unroll16": "tab:blue"}
print(f"  {'run':26} {'allpairs_med':>12} {'UT->T_med':>10} {'UT->T_p95':>10}  by-p-quartile")
for lab, run, kind in RUNS:
    m, nu = load(run)
    pred = predict(m)
    dpos = np.hypot(pred[:, 0] - Yt[:, 0], pred[:, 1] - Yt[:, 1]) * 1e3
    byq = [float(np.median(dpos[(Pt >= a) & (Pt < b)]))
           for a, b in zip([0, pq[0], pq[1], pq[2]], [pq[0], pq[1], pq[2], 1e9])]
    um, up95, be, ne = utt_metrics(run)
    res[run] = {"label": lab, "kind": kind, "n_unroll": nu,
                "kick_order": json.load(open(os.path.join(TM, run, "config.json"))).get("kick_order", 1),
                "allpairs_median_um": float(np.median(dpos)),
                "allpairs_p95_um": float(np.percentile(dpos, 95)),
                "allpairs_by_p_quartile_um": byq,
                "utt_median_um": um, "utt_p95_um": up95,
                "best_epoch": be, "epochs": ne}
    if kind == "single":
        c = single_colors[run]; ls = "--"
    elif kind == "curv":
        c = curv_colors[run]; ls = "-."
    else:
        c = unroll_cmap[run]; ls = "-"
    leg = f"{lab} — UT→T {um/1000:.2f} mm" if um else f"{lab}"
    ax.plot(cen, med_vs_qop(dpos), ls, color=c, lw=1.9, label=leg)
    print(f"  {run:26} {np.median(dpos):11.1f}u {('%.0f'%um) if um else '-':>9}u {('%.0f'%up95) if up95 else '-':>9}u  {['%.0f'%v for v in byq]}")

ax.axhline(3000, color="k", ls=":", lw=.8); ax.text(cen[0], 3100, "3 mm floor", fontsize=8)
ax.axhline(1000, color="g", ls=":", lw=.9); ax.text(cen[0], 1030, "1 mm target", fontsize=8, color="g")
ax.axhline(15, color="purple", ls="-.", lw=.8); ax.text(cen[0], 16, "extrapUTT ~15 µm", fontsize=8, color="purple")
ax.set_yscale("log"); ax.set_xlabel("signed q/p (qop)"); ax.set_ylabel("median |Δpos| [µm]")
ax.set_title("Multi-step unroll & 2nd-order curvature vs the single-shot floor (gen-4, fp32)\n"
             "curves = all-pairs median |Δpos| vs q/p · legend = UT→T-subset median (the headline)", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e1_vs_qop.png"), dpi=130); plt.close(fig)
json.dump({"n_eval": len(ti), "p_quartiles": pq.tolist(), "runs": res},
          open(os.path.join(RES, "e1_eval.json"), "w"), indent=2)
print("wrote figures/fig_e1_vs_qop.png + results/e1_eval.json")
