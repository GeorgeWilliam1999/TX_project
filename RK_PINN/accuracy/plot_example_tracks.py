#!/usr/bin/env python3
"""plot_example_tracks.py — visual confirmation of the label-fidelity finding.

For example tracks at low / mid / high momentum it shows, ALL in the gen-4
convention (the EOM the labels are built on, validated vs extrapUTT to 15 um):
  * the true trajectory x(z) through the v8r1 dipole (DOP853, adaptive fp64),
  * the UT->T endpoint as computed by three INDEPENDENT schemes:
        - 5 mm fixed-step RK4   = exactly how the gen-4 labels were made,
        - 1 mm fixed-step RK4   = 625x finer (RK4 error ~ h^4),
        - DOP853 adaptive       = a DIFFERENT method entirely (Dormand-Prince 8),
  * a zoom at the endpoint with a micron scale bar, annotated with the pairwise
    gaps. If the three coincide to ~um, the 5 mm labels are a converged solution
    of the EOM -> the ~3 mm surrogate floor is NOT a ground-truth problem.

'Truth' here is the accurately-integrated EOM (DOP853 / fine RK4) -- NOT the
deployed Allen RK (a buggy ~250 um approximation) and NOT extrapUTT (an ~11 um
approximation). Those are benchmarks, not references.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
CORPUS = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/data/train_10M_gen4.npz"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "core"))
from field_v8r1 import FieldV8R1  # noqa: E402

FIELD = FieldV8R1()
KAPPA, C_QP = 1.0e-3, 0.299792458


def deriv(S, z):  # vectorised, gen-4 EOM (identical to generate_data_v2.py)
    x, y, tx, ty, qop = S.T
    Bx, By, Bz = FIELD(x, y, z)
    k = KAPPA * qop
    N = np.sqrt(1 + tx * tx + ty * ty)
    return np.stack([tx, ty,
                     k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
                     k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
                     np.zeros_like(x)], axis=1)


def rk4_path(S0, z0, zf, step, record_mm=25.0):
    """Single-track RK4; returns endpoint and a (z,x,y) path sampled ~every record_mm."""
    S = S0.astype(np.float64).copy().reshape(1, 5)
    z = float(z0); dirn = 1.0 if zf > z0 else -1.0
    zs, xs, ys = [z], [S[0, 0]], [S[0, 1]]
    last = z
    while (zf - z) * dirn > 1e-7:
        h = dirn * min(step, abs(zf - z))
        zc = np.array([z])
        k1 = deriv(S, zc); k2 = deriv(S + 0.5 * h * k1, zc + 0.5 * h)
        k3 = deriv(S + 0.5 * h * k2, zc + 0.5 * h); k4 = deriv(S + h * k3, zc + h)
        S = S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4); z += h
        if abs(z - last) >= record_mm:
            zs.append(z); xs.append(S[0, 0]); ys.append(S[0, 1]); last = z
    zs.append(z); xs.append(S[0, 0]); ys.append(S[0, 1])
    return S[0].copy(), np.array(zs), np.array(xs), np.array(ys)


def dop853_endpoint(S0, z0, zf):
    qop = float(S0[4])
    def rhs(z, s):
        Sx = np.array([[s[0], s[1], s[2], s[3], qop]])
        return deriv(Sx, np.array([z]))[0, :4]
    sol = solve_ivp(rhs, (z0, zf), S0[:4].astype(float), method="DOP853",
                    rtol=1e-11, atol=1e-12, dense_output=True)
    return sol.y[:, -1], sol


def pick_examples():
    rng = np.random.default_rng(7)
    d = np.load(CORPUS, mmap_mode="r")
    N = d["X"].shape[0]
    idx = rng.choice(N, size=200000, replace=False)
    X = np.asarray(d["X"][np.sort(idx)], np.float64)
    P = np.asarray(d["P"][np.sort(idx)], np.float64)
    adz = np.abs(X[:, 6])
    forward = (X[:, 6] > 0)
    longish = (adz > 3500) & (adz < 7000) & forward
    out = []
    for label, plo, phi in [("low-p", 1.0, 2.0), ("mid-p", 12.0, 20.0), ("high-p", 80.0, 200.0)]:
        m = longish & (P >= plo) & (P < phi)
        cand = np.where(m)[0]
        # pick one with a sizeable transverse slope so pT is representative
        j = cand[np.argmax(np.abs(X[cand, 2]) + np.abs(X[cand, 3]))]
        out.append((label, X[j], P[j]))
    return out


def main():
    ex = pick_examples()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (label, x, p) in enumerate(ex):
        S0 = x[:5].copy(); z0, dz = x[5], x[6]; zf = z0 + dz
        tx, ty = x[2], x[3]
        pT = p * np.sqrt(tx * tx + ty * ty) / np.sqrt(1 + tx * tx + ty * ty)
        E5, _, _, _ = rk4_path(S0, z0, zf, 5.0)
        E1, zs, xs, ys = rk4_path(S0, z0, zf, 1.0)
        Edop, sol = dop853_endpoint(S0, z0, zf)
        # gaps (um) vs DOP853 truth
        g5 = np.hypot(E5[0] - Edop[0], E5[1] - Edop[1]) * 1e3
        g1 = np.hypot(E1[0] - Edop[0], E1[1] - Edop[1]) * 1e3
        bend_mm = abs(xs[-1] - (xs[0] + tx * dz))  # deviation from a straight line, mm

        ax = axes[0, col]
        ax.plot(zs, xs, "-", color="tab:blue", lw=1.6, label="true trajectory x(z)")
        ax.plot([z0, zf], [xs[0], xs[0] + tx * dz], "--", color="gray", lw=1.0,
                label="straight line (no field)")
        ax.scatter([z0], [xs[0]], c="k", s=20, zorder=5)
        ax.set_title(f"{label}:  p={p:.1f} GeV,  pT={pT:.2f} GeV\n|dz|={abs(dz):.0f} mm,  bend={bend_mm:.1f} mm")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("x [mm]"); ax.legend(fontsize=8); ax.grid(alpha=.3)

        # endpoint zoom (microns around DOP853 truth)
        az = axes[1, col]
        cx, cy = Edop[0], Edop[1]
        for E, name, mk, c in [(Edop, "DOP853 truth", "*", "k"),
                               (E5, "5 mm RK4 (label)", "o", "tab:red"),
                               (E1, "1 mm RK4", "x", "tab:green")]:
            az.scatter([(E[0] - cx) * 1e3], [(E[1] - cy) * 1e3], marker=mk, s=120, c=c, label=name, zorder=5)
        az.axhline(0, color="gray", lw=.5); az.axvline(0, color="gray", lw=.5)
        az.set_title(f"endpoint zoom — label gap {g5:.2f} µm\n(1 mm gap {g1:.3f} µm)")
        az.set_xlabel("x − x_truth [µm]"); az.set_ylabel("y − y_truth [µm]")
        az.legend(fontsize=8); az.grid(alpha=.3)
        lim = max(3.0, g5 * 1.6)
        az.set_xlim(-lim, lim); az.set_ylim(-lim, lim)
        print(f"{label}: p={p:.1f} GeV pT={pT:.2f} |dz|={abs(dz):.0f}mm bend={bend_mm:.1f}mm "
              f"| 5mm-vs-truth={g5:.2f}um  1mm-vs-truth={g1:.3f}um")

    fig.suptitle("Label fidelity: the 5 mm-RK4 gen-4 labels vs converged truth (DOP853 + 1 mm RK4), "
                 "gen-4 EOM / v8r1 field\nTop: true trajectory (bend vs straight line).  "
                 "Bottom: endpoint agreement at micron scale.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "label_fidelity_example_tracks.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
