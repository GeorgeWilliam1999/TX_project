#!/usr/bin/env python3
"""scan_eval_per_variable.py — accuracy vs (position, tx, ty, qop/p) for every scan run.

For each trained run: rebuild the model from its config.json, load best_model.pt +
normalization.json, run the held-out TEST split (test_indices.npy, identical across
runs at seed 42), and profile the endpoint error against each input variable.

Outputs (figures/, results/):
  fig_scan_acc_vs_variable_blockA.png   median |Δpos| vs p, |position|, tx, ty (Block A)
  fig_scan_acc_vs_variable_blockB.png   same (Block B)
  fig_scan_acc_channels_blockA/B.png    per-channel |Δtx|,|Δty| vs p
  scan_per_variable.json                overall + by-p-quartile medians per run

Runs on GPU if available, else CPU (test split ~520k tracks, tiny MLP — fast either way).
Skips runs not present yet.
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

BLOCK_A = [("d2 tanh", "scanA_d2_tanh"), ("d2 silu", "scanA_d2_silu"),
           ("d2 gelu", "scanA_d2_gelu"), ("d2 sin", "scanA_d2_sin"),
           ("d3 tanh", "scanA_d3_tanh"), ("d3 silu", "scanA_d3_silu"),
           ("d3 gelu", "scanA_d3_gelu"), ("d3 sin", "scanA_d3_sin")]
BLOCK_B = [("h96 baseline", "wave2_resid_h96"), ("alpha=0.5", "scanB_alpha05"),
           ("huber d=2", "scanB_delta2"), ("inv-p weight", "scanB_invp"),
           ("log-cosh", "scanB_logcosh")]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run):
    d = os.path.join(TM, run)
    if not os.path.exists(os.path.join(d, "best_model.pt")):
        return None
    c = json.load(open(os.path.join(d, "config.json")))
    m = create_model("pinn_v2", hidden_dims=c["hidden_dims"], activation=c["activation"],
                     dropout=c.get("dropout", 0.0), lambda_pde=c.get("lambda_pde", 0.0),
                     lambda_ic=c.get("lambda_ic", 0.0), n_collocation=c.get("n_collocation", 2),
                     kick_scaled_head=c.get("kick_scaled_head", False),
                     pde_scale_mode=c.get("pde_scale_mode", "legacy"),
                     pde_ref_length=c.get("pde_ref_length", 5161.0),
                     siren_w0=c.get("siren_w0", 30.0))
    if os.path.exists(os.path.join(d, "normalization.json")):
        m.load_normalization(os.path.join(d, "normalization.json"))
    ck = torch.load(os.path.join(d, "best_model.pt"), weights_only=False, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"]); m.eval().to(DEV)
    return m


@torch.no_grad()
def predict(m, X):
    out = np.empty((X.shape[0], 4), np.float64)
    for i in range(0, X.shape[0], 200000):
        xb = torch.from_numpy(X[i:i+200000].astype(np.float32)).to(DEV)
        out[i:i+200000] = m(xb).cpu().numpy()[:, :4]
    return out


def bin_median(var, err, edges):
    idx = np.digitize(var, edges) - 1
    cen, med = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 50:
            cen.append(0.5 * (edges[b] + edges[b+1])); med.append(np.median(err[sel]))
    return np.array(cen), np.array(med)


def main():
    with np.load(os.path.join(LAB, "data", "train_wave2_deploy.npz")) as d:
        X, Y, P = d["X"], d["Y"], d["P"]
    # one shared test split (seed 42 identical) — take from the first available run
    ti = None
    for _, run in BLOCK_A + BLOCK_B:
        f = os.path.join(TM, run, "test_indices.npy")
        if os.path.exists(f):
            ti = np.load(f); break
    if ti is None:
        print("no runs finished yet — nothing to eval"); return
    Xt, Yt, Pt = X[ti], Y[ti], P[ti]
    print(f"test split: {len(ti)} tracks  device={DEV}")

    # variable definitions: (key, values, edges, xlabel, logx)
    rpos = np.hypot(Xt[:, 0], Xt[:, 1])
    VARS = [
        ("p",  Pt,        np.logspace(np.log10(1.0), np.log10(200.0), 21), "momentum p [GeV]", True),
        ("rpos", rpos,    np.linspace(*np.percentile(rpos, [1, 99]), 21), "|position| √(x²+y²) [mm]", False),
        ("tx", Xt[:, 2],  np.linspace(*np.percentile(Xt[:, 2], [1, 99]), 21), "tx [rad]", False),
        ("ty", Xt[:, 3],  np.linspace(*np.percentile(Xt[:, 3], [1, 99]), 21), "ty [rad]", False),
    ]
    pq = np.percentile(Pt, [25, 50, 75])

    def eval_block(block):
        out = {}
        for lab, run in block:
            m = load_model(run)
            if m is None:
                print(f"  [skip] {run}"); continue
            pred = predict(m, Xt)
            dpos = np.hypot(pred[:, 0] - Yt[:, 0], pred[:, 1] - Yt[:, 1]) * 1e3  # µm
            dtx = np.abs(pred[:, 2] - Yt[:, 2]) * 1e3   # mrad
            dty = np.abs(pred[:, 3] - Yt[:, 3]) * 1e3
            byq = [float(np.median(dpos[(Pt >= a) & (Pt < b)]))
                   for a, b in zip([0, pq[0], pq[1], pq[2]], [pq[0], pq[1], pq[2], 1e9])]
            out[run] = {"label": lab, "dpos": dpos, "dtx": dtx, "dty": dty,
                        "median_um": float(np.median(dpos)), "p95_um": float(np.percentile(dpos, 95)),
                        "by_p_quartile_um": byq,
                        "median_dtx_mrad": float(np.median(dtx)),
                        "median_dty_mrad": float(np.median(dty))}
            print(f"  {run:<18} median={out[run]['median_um']:8.1f} µm  p95={out[run]['p95_um']:9.1f}  "
                  f"by-p={['%.0f'%v for v in byq]}")
        return out

    print("Block A:"); A = eval_block(BLOCK_A)
    print("Block B:"); B = eval_block(BLOCK_B)

    def plot_vars(block_res, title, fname):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9)); axes = axes.ravel()
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(block_res), 1)))
        for ax, (key, vals, edges, xlab, logx) in zip(axes, VARS):
            for (run, r), c in zip(block_res.items(), cmap):
                cen, med = bin_median(vals, r["dpos"], edges)
                if len(cen): ax.plot(cen, med, "o-", color=c, ms=3, lw=1.4, label=r["label"])
            if logx: ax.set_xscale("log")
            ax.set_yscale("log"); ax.set_xlabel(xlab); ax.set_ylabel("median |Δpos| [µm]")
            ax.axhline(3000, color="k", ls="--", lw=.6); ax.axhline(1000, color="g", ls=":", lw=.6)
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, ncol=2)
        fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(FIG, fname), dpi=130); plt.close(fig); print("wrote", fname)

    def plot_channels(block_res, title, fname):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(block_res), 1)))
        for ax, ch, lab2 in [(axes[0], "dtx", "|Δtx| [mrad]"), (axes[1], "dty", "|Δty| [mrad]")]:
            for (run, r), c in zip(block_res.items(), cmap):
                cen, med = bin_median(Pt, r[ch], VARS[0][2])
                if len(cen): ax.plot(cen, med, "o-", color=c, ms=3, lw=1.4, label=r["label"])
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("momentum p [GeV]")
            ax.set_ylabel("median " + lab2); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, ncol=2)
        fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(FIG, fname), dpi=130); plt.close(fig); print("wrote", fname)

    if A:
        plot_vars(A, "Scan Block A — accuracy vs track variable (depth × activation)", "fig_scan_acc_vs_variable_blockA.png")
        plot_channels(A, "Scan Block A — slope-channel error vs momentum", "fig_scan_acc_channels_blockA.png")
    if B:
        plot_vars(B, "Scan Block B — accuracy vs track variable (cost function)", "fig_scan_acc_vs_variable_blockB.png")
        plot_channels(B, "Scan Block B — slope-channel error vs momentum", "fig_scan_acc_channels_blockB.png")

    # JSON (drop the big per-track arrays)
    def strip(res):
        return {k: {kk: vv for kk, vv in v.items() if kk not in ("dpos", "dtx", "dty")}
                for k, v in res.items()}
    json.dump({"device": DEV, "n_test": int(len(ti)), "p_quartiles": pq.tolist(),
               "blockA": strip(A), "blockB": strip(B)},
              open(os.path.join(RES, "scan_per_variable.json"), "w"), indent=2)
    print("wrote results/scan_per_variable.json")


if __name__ == "__main__":
    main()
