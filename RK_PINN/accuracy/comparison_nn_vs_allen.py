#!/usr/bin/env python3
"""comparison_nn_vs_allen.py — head-to-head ACCURACY: our NN vs Allen's extrapolators.

Common ground: the fixed UT->T plane (z 2665 -> 7826), the set where the production
polynomial scores ~15 um median. Truth Y_true is the gen-4 RK4 (== DOP853 to <0.2 um,
proven by label_fidelity.py). Convention verified identical across gen-4 / harness / Allen
(field loaders differ x1000, the 1e-3 cancels; endpoints agree to 0.0 mm).

Methods scored vs truth (all Allen-faithful, file:line-grounded in Allen_scoping/harness):
  extrapUTT (Allen fast UT->T polynomial)   -- precomputed plane_poly_v8r1_polm1.csv
  Allen RK Cash-Karp deployed (100 mm, bug) -- rk_allen_cashkarp(buggy=True)  [extrapolate_states_t]
  Allen RK Cash-Karp corrected (100 mm)     -- rk_allen_cashkarp(buggy=False)
  Allen Nystrom fast-step (ttrack chain)    -- rk_nystrom_fast (the most up-to-date fast integrator)
  NN deployed candidate (h96)               -- candidate/pinn_v2_ALLEN_v1
  NN h64 (accuracy-equivalent)              -- wave2_resid_h64 (if present)
  straight line (control)
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

REPO = "/data/bfys/gscriven/track-extrapolation-pinn"
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"
PAPER = os.path.join(LAB, "paper_p0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "models"))
sys.path.insert(0, "/data/bfys/gscriven/Ex_rep/Allen_scoping/harness")

import torch  # noqa: E402
from architectures import create_model  # noqa: E402
import fieldmap as fm, integrators as ig  # noqa: E402

ZINI, ZFIN = 2665.0, 7826.0
DZ = ZFIN - ZINI


def load_model_dir(d):
    ckpt = torch.load(os.path.join(d, "best_model.pt"), weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    m = create_model("pinn_v2", hidden_dims=cfg["hidden_dims"], activation=cfg["activation"],
                     dropout=cfg.get("dropout", 0.0), lambda_pde=cfg.get("lambda_pde", 0.0),
                     lambda_ic=cfg.get("lambda_ic", 0.0), n_collocation=cfg.get("n_collocation", 2),
                     kick_scaled_head=cfg.get("kick_scaled_head", False),
                     pde_scale_mode=cfg.get("pde_scale_mode", "legacy"),
                     pde_ref_length=cfg.get("pde_ref_length", 5161.0))
    nj = os.path.join(d, "normalization.json")
    if os.path.exists(nj):
        m.load_normalization(nj)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m


def metrics(pred, Y, p):
    dx = np.abs(pred[:, 0] - Y[:, 0]) * 1e3          # um (bending plane)
    dpos = np.hypot(pred[:, 0] - Y[:, 0], pred[:, 1] - Y[:, 1]) * 1e3  # um
    dtx = np.abs(pred[:, 2] - Y[:, 2]) * 1e6          # urad
    edges = np.quantile(p, [0.25, 0.5, 0.75])
    b = np.digitize(p, edges)  # 0=low p .. 3=high p
    byq_lo2hi = [float(np.median(dpos[b == k])) for k in range(4)]
    f = lambda a, q: float(np.quantile(a, q))
    return dict(median_dx_um=float(np.median(dx)), median_pos_um=float(np.median(dpos)),
                p68_pos_um=f(dpos, .68), p95_pos_um=f(dpos, .95), p99_pos_um=f(dpos, .99),
                max_pos_um=float(dpos.max()), median_dtx_urad=float(np.median(dtx)),
                median_pos_um_by_p_quartile_lo2hi=byq_lo2hi), dpos


def main():
    n_int = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    d = np.load(os.path.join(PAPER, "plane_ref_v8r1.npz"))
    X, Y = np.asarray(d["X_plane"], np.float64), np.asarray(d["Y_true"], np.float64)
    poly = np.loadtxt(os.path.join(PAPER, "plane_poly_v8r1_polm1.csv"), delimiter=",", skiprows=1)
    qop = X[:, 4]; p = 0.299792458 / np.abs(qop)
    Ntot = X.shape[0]
    rng = np.random.default_rng(20260623)
    sub = np.sort(rng.choice(Ntot, size=min(n_int, Ntot), replace=False))
    print(f"plane-ref: {Ntot} tracks, p[{p.min():.1f},{p.max():.0f}] GeV; integrator subsample n={len(sub)}")

    arrays, results = {"p_GeV": p[sub]}, {}

    # --- Allen integrators (harness, Allen-faithful), per-track on the subsample ---
    FH = fm.FieldMap()
    def run_int(fn, **kw):
        out = np.empty((len(sub), 4))
        for i, j in enumerate(sub):
            out[i] = fn(FH, ZINI, ZFIN, X[j, :4], X[j, 4], **kw)
        return out
    t0 = time.time()
    e_ckb = run_int(ig.rk_allen_cashkarp, dz=100.0, buggy=True)
    e_ckc = run_int(ig.rk_allen_cashkarp, dz=100.0, buggy=False)
    e_nys = run_int(ig.rk_nystrom_fast, step=500.0)
    print(f"Allen integrators ({len(sub)} tracks x3) in {time.time()-t0:.1f}s")
    # convention cross-check: harness DOP853 vs the stored truth on a small subset
    chk = sub[:200]
    dop = np.array([ig.truth_endpoint(FH, ZINI, ZFIN, X[j, :4], X[j, 4]) for j in chk])
    cc = np.hypot(dop[:, 0] - Y[chk, 0], dop[:, 1] - Y[chk, 1]) * 1e3
    print(f"convention cross-check: harness DOP853 vs stored truth = {np.median(cc):.3f} um median (≈0 expected)")

    Ysub, psub = Y[sub], p[sub]
    results["extrapUTT (Allen fast UT->T)"], arrays["extrapUTT"] = metrics(poly[sub], Ysub, psub)
    results["Allen RK Cash-Karp (deployed, bug)"], arrays["RK_deployed"] = metrics(e_ckb, Ysub, psub)
    results["Allen RK Cash-Karp (corrected)"], arrays["RK_corrected"] = metrics(e_ckc, Ysub, psub)
    results["Allen Nystrom fast-step"], arrays["Nystrom"] = metrics(e_nys, Ysub, psub)
    sl = np.stack([X[:, 0] + X[:, 2] * DZ, X[:, 1] + X[:, 3] * DZ, X[:, 2], X[:, 3]], axis=1)
    results["straight line (control)"], arrays["straight_line"] = metrics(sl[sub], Ysub, psub)

    # --- NN(s) ---
    Xin = np.concatenate([X[:, :5], np.full((Ntot, 1), ZINI), np.full((Ntot, 1), DZ)], axis=1).astype(np.float32)
    nn_dirs = {"NN deployed candidate (h96)": os.path.join(REPO, "candidate", "pinn_v2_ALLEN_v1"),
               "NN h64 (accuracy-equiv)": os.path.join(LAB, "trained_models", "wave2_resid_h64"),
               "NN h96 (wave2)": os.path.join(LAB, "trained_models", "wave2_resid_h96")}
    with torch.no_grad():
        for name, dd in nn_dirs.items():
            if not os.path.exists(os.path.join(dd, "best_model.pt")):
                print(f"SKIP {name} (no ckpt)"); continue
            pred = load_model_dir(dd)(torch.from_numpy(Xin)).numpy().astype(np.float64)
            results[name], arrays[name] = metrics(pred[sub], Ysub, psub)

    # --- report ---
    print(f"\n{'method':<38}{'med pos':>9}{'med dx':>8}{'p95':>9}{'p99':>9}{'max':>9}  byP lo->hi (um)")
    print("-" * 110)
    order = ["NN deployed candidate (h96)", "NN h64 (accuracy-equiv)", "NN h96 (wave2)",
             "Allen Nystrom fast-step", "Allen RK Cash-Karp (deployed, bug)",
             "extrapUTT (Allen fast UT->T)", "Allen RK Cash-Karp (corrected)", "straight line (control)"]
    for k in order:
        if k not in results: continue
        m = results[k]
        bq = "/".join(f"{v:.0f}" for v in m["median_pos_um_by_p_quartile_lo2hi"])
        print(f"{k:<38}{m['median_pos_um']:>9.1f}{m['median_dx_um']:>8.1f}{m['p95_pos_um']:>9.1f}"
              f"{m['p99_pos_um']:>9.1f}{m['max_pos_um']:>9.0f}  [{bq}]")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    blob = {"set": "plane_ref UT->T (z 2665->7826)", "n_total": int(Ntot), "n_integrator_subsample": int(len(sub)),
            "truth": "gen-4 RK4 (Y_true) == DOP853 to <0.2 um", "convention_crosscheck_um": float(np.median(cc)),
            "metric": "median |dpos| (um) primary; |dx| matches the project's median_dx_um",
            "gpu_speed_ns_per_track": {"extrapUTT": 2.34, "RK_field_cashkarp": 5.71,
                                       "NN_baseline_h96": 7.05, "NN_fused_h96_bitexact": 4.85,
                                       "NN_h64_fu": 0.91},
            "results": results}
    json.dump(blob, open(os.path.join(HERE, "results", "comparison_nn_vs_allen.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(HERE, "results", "comparison_arrays.npz"), **arrays)
    print("\nwrote results/comparison_nn_vs_allen.json + comparison_arrays.npz")


if __name__ == "__main__":
    main()
