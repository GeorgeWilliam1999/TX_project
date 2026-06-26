#!/usr/bin/env python3
"""rk4_bar.py — the field-access accuracy bar (E5), on the real gen-4 field.

Classical RK4 of the Lorentz ODE (NeuralRK4 with disable_correction=True — no
learned term, no training) at n_rk_steps in {1,2,4,8}, evaluated on the shared
seed-42 test split. Gives what an integrator WITH field access achieves on v8r1
— the bar the pure-NN (~3 mm floor) experiments are measured against.

Outputs: figures/fig_rk4_bar_vs_qop.png, results/rk4_bar.json
Same metric/bins as analysis/scan_arch_x_cost so it overlays the NN floor.
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

N_SUB = 20_000   # CPU subsample (field lookups dominate; convergence sweep is costly)
STEPS = [8, 32, 128, 512]   # classical RK4 needs many steps over the ~5 m UT->T crossing
                            # (truth = 5 mm step ~ 1000); small n is under-resolved / diverges

with np.load(os.path.join(LAB, "data", "train_wave2_deploy.npz")) as d:
    X, Y, P = d["X"], d["Y"], d["P"]
ti = np.load(os.path.join(TM, "scanA_d2_tanh", "test_indices.npy"))
rng = np.random.default_rng(42); ti = rng.choice(ti, size=min(N_SUB, len(ti)), replace=False)
Xt, Yt, Pt = X[ti], Y[ti], P[ti]
qop = Xt[:, 4].astype(np.float64)
k = float(np.median(np.abs(qop) * Pt))
print(f"bar eval on {len(ti)} tracks (CPU)")

edges = np.linspace(*np.percentile(qop, [0.5, 99.5]), 41)
cen = 0.5 * (edges[:-1] + edges[1:])
pq = np.percentile(Pt, [25, 50, 75])


def med_vs_qop(dpos):
    idx = np.digitize(qop, edges) - 1
    out = np.full(len(cen), np.nan)
    for b in range(len(cen)):
        sel = idx == b
        if sel.sum() >= 50:
            out[b] = np.nanmedian(dpos[sel])
    return out


@torch.no_grad()
def run_rk4(n):
    m = create_model("neural_rk4", hidden_dims=[64, 64], activation="tanh",
                     n_rk_steps=n, disable_correction=True).eval()
    pred = np.empty((len(ti), 5), np.float64)
    for i in range(0, len(ti), 100000):
        xb = torch.from_numpy(Xt[i:i+100000].astype(np.float32))
        pred[i:i+100000] = m(xb).numpy()
    dpos = np.hypot(pred[:, 0] - Yt[:, 0], pred[:, 1] - Yt[:, 1]) * 1e3  # µm
    byq = [float(np.nanmedian(dpos[(Pt >= a) & (Pt < b)]))
           for a, b in zip([0, pq[0], pq[1], pq[2]], [pq[0], pq[1], pq[2], 1e9])]
    nan_frac = float(np.mean(~np.isfinite(dpos)))
    return dpos, byq, nan_frac


res = {}
fig, ax = plt.subplots(figsize=(9, 6.5))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(STEPS)))
for n, c in zip(STEPS, cmap):
    dpos, byq, nan_frac = run_rk4(n)
    res[f"n{n}"] = {"median_um": float(np.nanmedian(dpos)), "p95_um": float(np.nanpercentile(dpos, 95)),
                    "by_p_quartile_um": byq, "nan_frac": nan_frac}
    print(f"  RK4 n={n:>4}: median={np.nanmedian(dpos):9.2f} µm  p95={np.nanpercentile(dpos,95):10.1f}  "
          f"diverged={nan_frac:.1%}  by-p={['%.1f'%v for v in byq]}")
    ax.plot(cen, med_vs_qop(dpos), "o-", color=c, ms=3, lw=1.6, label=f"RK4 n={n} (diverged {nan_frac:.0%})")
ax.axhline(3000, color="k", ls="--", lw=.8); ax.text(cen[0], 3300, "NN ~3 mm floor", fontsize=8)
ax.axhline(15, color="purple", ls="-.", lw=.8); ax.text(cen[0], 16, "extrapUTT ~15 µm", fontsize=8, color="purple")
ax.set_yscale("log"); ax.set_xlabel("signed q/p (qop)"); ax.set_ylabel("median |Δpos| [µm]")
ax.set_title("Field-access bar: classical RK4 vs the pure-NN floor (real gen-4 field)", fontsize=11)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rk4_bar_vs_qop.png"), dpi=130); plt.close(fig)
json.dump({"n_eval": len(ti), "p_quartiles": pq.tolist(), "steps": res},
          open(os.path.join(RES, "rk4_bar.json"), "w"), indent=2)
print("wrote figures/fig_rk4_bar_vs_qop.png + results/rk4_bar.json")
