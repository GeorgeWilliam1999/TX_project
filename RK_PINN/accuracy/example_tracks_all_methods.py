#!/usr/bin/env python3
"""example_tracks_all_methods.py — concrete per-track view: where does each method's
UT->T endpoint land vs truth, for a low-p and a high-p track. Trajectory = truth
(DOP853, gen-4 EOM); endpoint markers: truth, extrapUTT, NN h64, Allen Nystrom,
Allen RK deployed. All convention-consistent (verified to 0.001 um)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"
PAPER = os.path.join(LAB, "paper_p0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "models"))
sys.path.insert(0, os.path.join(REPO, "core"))
sys.path.insert(0, "/data/bfys/gscriven/Ex_rep/Allen_scoping/harness")
import torch
from architectures import create_model
from field_v8r1 import FieldV8R1
import fieldmap as fm, integrators as ig

F4 = FieldV8R1(); KAPPA = 1e-3
ZINI, ZFIN = 2665.0, 7826.0


def deriv(S, z):
    x, y, tx, ty, qop = S.T; Bx, By, Bz = F4(x, y, z); k = KAPPA * qop
    N = np.sqrt(1 + tx * tx + ty * ty)
    return np.stack([tx, ty, k*N*(tx*ty*Bx-(1+tx*tx)*By+ty*Bz),
                     k*N*((1+ty*ty)*Bx-tx*ty*By-tx*Bz), np.zeros_like(x)], 1)


def truth_path(S0, z0, zf, npts=200):
    qop = float(S0[4])
    sol = solve_ivp(lambda z, s: deriv(np.array([[s[0], s[1], s[2], s[3], qop]]), np.array([z]))[0, :4],
                    (z0, zf), S0[:4], method="DOP853", rtol=1e-10, atol=1e-11, dense_output=True)
    zs = np.linspace(z0, zf, npts); P = sol.sol(zs)
    return zs, P[0], P[1], sol.y[:, -1]


def load_nn(d):
    ck = torch.load(os.path.join(d, "best_model.pt"), weights_only=False, map_location="cpu")
    c = ck["config"]; m = create_model("pinn_v2", hidden_dims=c["hidden_dims"], activation=c["activation"],
        dropout=c.get("dropout", 0.0), lambda_pde=c.get("lambda_pde", 0.0), lambda_ic=c.get("lambda_ic", 0.0),
        n_collocation=c.get("n_collocation", 2), kick_scaled_head=c.get("kick_scaled_head", False),
        pde_scale_mode=c.get("pde_scale_mode", "legacy"), pde_ref_length=c.get("pde_ref_length", 5161.0))
    nj = os.path.join(d, "normalization.json")
    if os.path.exists(nj): m.load_normalization(nj)
    m.load_state_dict(ck["model_state_dict"]); m.eval(); return m


def main():
    d = np.load(os.path.join(PAPER, "plane_ref_v8r1.npz"))
    X, Y = np.asarray(d["X_plane"], np.float64), np.asarray(d["Y_true"], np.float64)
    poly = np.loadtxt(os.path.join(PAPER, "plane_poly_v8r1_polm1.csv"), delimiter=",", skiprows=1)
    p = 0.299792458 / np.abs(X[:, 4])
    nn = load_nn(os.path.join(LAB, "trained_models", "wave2_resid_h64"))
    FH = fm.FieldMap()

    # NN error for all tracks -> pick a MEDIAN-REPRESENTATIVE track in each p-bin
    Xin_all = np.concatenate([X[:, :5], np.full((X.shape[0], 1), ZINI),
                              np.full((X.shape[0], 1), ZFIN - ZINI)], axis=1).astype(np.float32)
    with torch.no_grad():
        nn_all = nn(torch.from_numpy(Xin_all)).numpy()
    nn_err = np.hypot(nn_all[:, 0] - Y[:, 0], nn_all[:, 1] - Y[:, 1])

    def representative(mask):
        idx = np.where(mask)[0]
        med = np.median(nn_err[idx])
        return idx[np.argmin(np.abs(nn_err[idx] - med))]
    lo = representative((p > 1.8) & (p < 2.5))
    hi = representative((p > 60) & (p < 120))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (j, lab) in enumerate([(lo, "low-p"), (hi, "high-p")]):
        S0 = X[j, :5]; qop = X[j, 4]; pj = p[j]
        zs, xs, ys, truth = truth_path(S0, ZINI, ZFIN)
        Xin = np.concatenate([X[j, :5], [ZINI, ZFIN - ZINI]]).astype(np.float32)[None, :]
        with torch.no_grad():
            ennp = nn(torch.from_numpy(Xin)).numpy()[0]
        methods = [("truth", truth, "k", "*"),
                   ("extrapUTT (Allen)", poly[j], "tab:green", "P"),
                   ("Allen Nystrom", ig.rk_nystrom_fast(FH, ZINI, ZFIN, S0[:4], qop), "tab:cyan", "D"),
                   ("Allen RK deployed", ig.rk_allen_cashkarp(FH, ZINI, ZFIN, S0[:4], qop, dz=100, buggy=True), "tab:blue", "s"),
                   ("NN h64 (ours)", ennp, "tab:red", "o")]
        ax = axes[row, 0]
        ax.plot(zs, xs, "-", color="k", lw=1.3, label="true trajectory x(z)")
        ax.plot([ZINI, ZFIN], [xs[0], xs[0] + S0[2] * (ZFIN - ZINI)], "--", color="gray", lw=1, label="straight line")
        ax.scatter([ZINI], [xs[0]], c="k", s=20)
        ax.set_title(f"{lab}: p={pj:.1f} GeV  (bend {abs(xs[-1]-(xs[0]+S0[2]*(ZFIN-ZINI))):.0f} mm)")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("x [mm]"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        az = axes[row, 1]; cx, cy = truth[0], truth[1]
        for name, e, c, mk in methods:
            dxmm = (e[0] - cx); dymm = (e[1] - cy)
            az.scatter([dxmm], [dymm], c=c, marker=mk, s=140, edgecolor="k", zorder=5,
                       label=f"{name}: {np.hypot(dxmm,dymm)*1e3:.0f} µm" if np.hypot(dxmm,dymm)<1 else f"{name}: {np.hypot(dxmm,dymm):.1f} mm")
        az.axhline(0, color="gray", lw=.5); az.axvline(0, color="gray", lw=.5)
        az.set_title(f"{lab}: endpoint error vs truth (x–x_truth, y–y_truth)")
        az.set_xlabel("Δx [mm]"); az.set_ylabel("Δy [mm]"); az.legend(fontsize=8); az.grid(alpha=.3)
    fig.suptitle("Where each method lands at the T-plane vs truth — Allen Nyström & extrapUTT hit close;\n"
                 "our NN misses by ~mm (worst at low p). Trajectory + per-method endpoint error.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "figures", "fig_example_tracks_all_methods.png"), dpi=130)
    print("wrote figures/fig_example_tracks_all_methods.png")


if __name__ == "__main__":
    main()
