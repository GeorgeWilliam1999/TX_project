#!/usr/bin/env python3
"""label_fidelity.py — diagnostic step 1a: how noisy are the gen-4 labels?

The gen-4 corpus targets are produced by classic RK4 at a FIXED 5 mm step
(datagen/generate_data_v2.py, KAPPA=1e-3, v8r1 field). If that 5 mm
discretisation is itself ~mm wrong (especially for low-p / long-dz tracks where
the bend is sharp), then NO model trained on those labels can beat ~mm — the
~3 mm surrogate floor would be (partly) LABEL-limited, not a model problem.

Method (zero convention risk): use the IDENTICAL RHS as the generator and only
vary the integrator step. Truth = the same RK4 at 1 mm (RK4 global error ∝ h^4,
so 1 mm is ~625× more accurate than 5 mm — the gap 5mm−1mm ≈ the 5 mm label
error). Cross-checks: (i) reproduce the stored 5 mm labels exactly; (ii) confirm
1 mm is converged against 0.5 mm on a subsample.

Outputs: results/label_fidelity.json + a printed summary, stratified by momentum
(the low-p tail is the Kalman-critical regime) and by |dz|.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
CORPUS = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/data/train_10M_gen4.npz"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "core"))
from field_v8r1 import FieldV8R1  # noqa: E402

FIELD = FieldV8R1()
KAPPA = 1.0e-3
C_QP = 0.299792458


def deriv(S, z):
    """Identical EOM to datagen/generate_data_v2.py::deriv, z per-track."""
    x, y, tx, ty, qop = S.T
    Bx, By, Bz = FIELD(x, y, z)
    k = KAPPA * qop
    N = np.sqrt(1 + tx * tx + ty * ty)
    return np.stack([tx, ty,
                     k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
                     k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
                     np.zeros_like(x)], axis=1)


def rk4_vec(S0, z0, zf, step, max_iter=200000):
    """Vectorised classic RK4 with per-track start/end planes (fixed step)."""
    S = S0.astype(np.float64).copy()
    z = z0.astype(np.float64).copy()
    zf = zf.astype(np.float64)
    dirn = np.sign(zf - z0)
    for _ in range(max_iter):
        rem = zf - z
        active = np.abs(rem) > 1e-7
        if not active.any():
            break
        h = np.where(active, dirn * np.minimum(step, np.abs(rem)), 0.0)
        hcol = h[:, None]
        k1 = deriv(S, z)
        k2 = deriv(S + 0.5 * hcol * k1, z + 0.5 * h)
        k3 = deriv(S + 0.5 * hcol * k2, z + 0.5 * h)
        k4 = deriv(S + hcol * k3, z + h)
        S = S + (hcol / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z = z + h
    return S


def pct(a):
    a = np.asarray(a, np.float64)
    p = np.percentile(a, [50, 68, 95, 99, 100])
    return {"median": float(p[0]), "p68": float(p[1]), "p95": float(p[2]),
            "p99": float(p[3]), "max": float(p[4]), "mean": float(a.mean())}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = 20260623
    rng = np.random.default_rng(seed)
    d = np.load(CORPUS, mmap_mode="r")
    N = d["X"].shape[0]
    idx = np.sort(rng.choice(N, size=n, replace=False))
    X = np.asarray(d["X"][idx], np.float64)   # [n,7] x,y,tx,ty,qop,z0,dz
    Y = np.asarray(d["Y"][idx], np.float64)   # [n,5] x1,y1,tx1,ty1,qop
    P = np.asarray(d["P"][idx], np.float64)   # [n] momentum GeV
    S0 = X[:, :5].copy()
    z0 = X[:, 5].copy()
    dz = X[:, 6].copy()
    zf = z0 + dz
    print(f"sampled {n} tracks | p[GeV] med={np.median(P):.1f} "
          f"({P.min():.2f}–{P.max():.0f}) | |dz| med={np.median(np.abs(dz)):.0f} "
          f"max={np.abs(dz).max():.0f} mm")

    t0 = time.time()
    S5 = rk4_vec(S0, z0, zf, 5.0)     # reproduce the labels
    S1 = rk4_vec(S0, z0, zf, 1.0)     # truth (625x more accurate than 5 mm)
    print(f"integrated 5mm + 1mm in {time.time()-t0:.1f}s")

    # cross-check (i): my 5 mm reproduces the stored labels (field/EOM match)
    repro = np.sqrt((S5[:, 0] - Y[:, 0])**2 + (S5[:, 1] - Y[:, 1])**2) * 1e3  # um
    # cross-check (ii): 1 mm converged vs 0.5 mm on a subsample
    sub = rng.choice(n, size=min(200, n), replace=False)
    Shalf = rk4_vec(S0[sub], z0[sub], zf[sub], 0.5)
    conv = np.sqrt((S1[sub, 0] - Shalf[:, 0])**2 + (S1[sub, 1] - Shalf[:, 1])**2) * 1e3  # um

    # label error = stored labels (5 mm) vs truth (1 mm), position in microns
    err_um = np.sqrt((Y[:, 0] - S1[:, 0])**2 + (Y[:, 1] - S1[:, 1])**2) * 1e3
    err_tx_urad = np.abs(Y[:, 2] - S1[:, 2]) * 1e6
    err_ty_urad = np.abs(Y[:, 3] - S1[:, 3]) * 1e6

    # momentum quartiles (hi→lo) and |dz| bins
    qedge = np.quantile(P, [0, .25, .5, .75, 1.0])
    pq = {}
    for i in range(4):
        m = (P >= qedge[i]) & (P <= qedge[i+1] if i == 3 else P < qedge[i+1])
        pq[f"p_q{i+1} [{qedge[i]:.1f}-{qedge[i+1]:.1f}]GeV"] = pct(err_um[m])
    dzb = {}
    for lo, hi in [(0, 300), (300, 1000), (1000, 3000), (3000, 11000)]:
        m = (np.abs(dz) >= lo) & (np.abs(dz) < hi)
        if m.sum():
            dzb[f"|dz|[{lo}-{hi}]mm n={int(m.sum())}"] = pct(err_um[m])
    # low-p (Kalman-critical) and the worst tracks
    lowp = pct(err_um[P < 12.0])

    out = {
        "diagnostic": "label_fidelity (5mm-RK4 labels vs 1mm-RK4 truth, identical RHS)",
        "n_tracks": n, "seed": seed,
        "crosscheck_5mm_reproduces_stored_labels_um": pct(repro),
        "crosscheck_1mm_vs_0p5mm_truth_converged_um": pct(conv),
        "label_pos_error_um_overall": pct(err_um),
        "label_pos_error_um_lowp_below12GeV": lowp,
        "label_pos_error_um_by_momentum_quartile": pq,
        "label_pos_error_um_by_dz": dzb,
        "label_slope_error_urad": {"tx": pct(err_tx_urad), "ty": pct(err_ty_urad)},
        "reference": {"surrogate_floor_mm": 3.0, "extrapUTT_median_um": 10.9,
                      "extrapUTT_lowp_um": 748.0, "extrapUTT_p99_um": 11738.0},
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "label_fidelity.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== LABEL FIDELITY (position error of the 5 mm labels vs truth) ===")
    print(f"  cross-check: my 5mm reproduces stored labels to {out['crosscheck_5mm_reproduces_stored_labels_um']['median']:.3f} um median (must be ~0)")
    print(f"  cross-check: 1mm vs 0.5mm truth agree to {out['crosscheck_1mm_vs_0p5mm_truth_converged_um']['median']:.4f} um median (truth converged)")
    o = out["label_pos_error_um_overall"]
    print(f"  OVERALL label error: median={o['median']:.2f} um  p95={o['p95']:.1f}  p99={o['p99']:.1f}  max={o['max']:.1f} um")
    print(f"  low-p (<12 GeV):     median={lowp['median']:.2f} um  p95={lowp['p95']:.1f}  p99={lowp['p99']:.1f}  max={lowp['max']:.1f} um")
    print("  by momentum quartile (hi→lo):")
    for k, v in pq.items():
        print(f"    {k:24s} median={v['median']:8.2f}  p95={v['p95']:9.1f}  max={v['max']:9.1f} um")
    print("  by |dz|:")
    for k, v in dzb.items():
        print(f"    {k:24s} median={v['median']:8.2f}  p95={v['p95']:9.1f}  max={v['max']:9.1f} um")
    print(f"\n  vs surrogate floor 3000 um, extrapUTT 10.9 um median / 748 um low-p / 11738 um p99")
    print("wrote results/label_fidelity.json")


if __name__ == "__main__":
    main()
