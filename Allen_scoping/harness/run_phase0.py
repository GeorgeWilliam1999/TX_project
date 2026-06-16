#!/usr/bin/env python3
"""
run_phase0.py -- Phase-0 foundation driver (rebuilt from source 2026-06-16).

Wires the harness together and prints/saves a report:
  1. kappa-guard self-tests (fail loud).
  2. DEPLOYED incumbent vs truth: Allen's buggy Cash-Karp (dz=100mm, fp32) vs
     DOP853, paired + stratified by momentum, with bootstrap CI.
  3. Bug cost: deployed (buggy) vs corrected Cash-Karp -- how much the
     off-by-one stage loop costs in accuracy.
  4. Step-size sweep of the incumbent: the throughput lever (dz vs accuracy vs
     field-evals/track).
  5. fp32 noise floor; Jacobian-gate demo.

Nothing here writes into the READ-ONLY Allen / TE_stack trees.
Run:  python3 run_phase0.py
"""
from __future__ import annotations
import json, time, os
import numpy as np

from fieldmap import FieldMap, C_LIGHT
import selftests
from integrators import (truth_endpoint, rk_allen_cashkarp, truth_jacobian, frob_rel)
import scorer
from sampler import sample_general

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRACKS = 120
DZ_DEPLOY = 100.0                 # extrapolate_states_t default step (mm)
DZ_SWEEP = [50.0, 100.0, 200.0, 400.0]
RNG = np.random.default_rng(7)


def _pos_um(resid):
    return np.hypot(resid[:, 0], resid[:, 1]) * scorer.POS_TO_UM


def _run_all(field, data, dz, buggy, dtype):
    return np.array([rk_allen_cashkarp(field, data["z0"][i], data["z1"][i], data["s0"][i],
                                       data["qop"][i], dz=dz, buggy=buggy, dtype=dtype)
                     for i in range(N_TRACKS)])


def main():
    t0 = time.time()
    field = FieldMap()
    print("=" * 74)
    print("PHASE 0 (rebuilt from source) -- truth + DEPLOYED incumbent + kappa-guard")
    print("=" * 74)

    selftests.run(field)

    # --- truth ------------------------------------------------------------
    print(f"\n{N_TRACKS} general-step tracks; truth = DOP853 (fp64)")
    data = sample_general(N_TRACKS, RNG)
    a = time.time()
    truth = np.array([truth_endpoint(field, data["z0"][i], data["z1"][i], data["s0"][i], data["qop"][i])
                      for i in range(N_TRACKS)])
    print(f"  truth: {time.time()-a:.1f}s")

    # --- 2. deployed incumbent (buggy Cash-Karp, dz=100, fp32) vs truth ---
    inc = _run_all(field, data, DZ_DEPLOY, buggy=True, dtype=np.float32)
    resid = inc - truth
    print(f"\nDEPLOYED incumbent (buggy Cash-Karp, dz={DZ_DEPLOY:.0f}mm, fp32) vs truth:")
    rows = scorer.stratify_by_momentum(data["qop"], resid)
    scorer.print_table(rows)
    med, lo, hi = scorer.bootstrap_ci(_pos_um(resid))
    print(f"  headline pos median = {med:.2f} um  (95% CI [{lo:.2f}, {hi:.2f}])")

    # --- 3. bug cost: buggy vs corrected Cash-Karp (same dz, fp32) --------
    cor = _run_all(field, data, DZ_DEPLOY, buggy=False, dtype=np.float32)
    buggy_med = float(np.percentile(_pos_um(resid), 50))
    buggy_p95 = float(np.percentile(_pos_um(resid), 95))
    cor_med = float(np.percentile(_pos_um(cor - truth), 50))
    cor_p95 = float(np.percentile(_pos_um(cor - truth), 95))
    print(f"\nOff-by-one bug cost at dz={DZ_DEPLOY:.0f}mm (pos vs truth, um):")
    print(f"  {'':12}{'median':>10}{'p95':>10}")
    print(f"  {'buggy (deployed)':12}{buggy_med:>10.2f}{buggy_p95:>10.2f}")
    print(f"  {'corrected':12}{cor_med:>10.2f}{cor_p95:>10.2f}")
    print(f"  -> the deployed bug inflates the median error ~{buggy_med/max(cor_med,1e-9):.0f}x")

    # --- 4. step-size sweep of the incumbent (throughput lever) -----------
    print("\nIncumbent step-size sweep (buggy Cash-Karp, fp32): dz vs accuracy vs cost")
    print(f"  {'dz[mm]':>7} {'evals/trk':>10} {'pos.med':>9} {'pos.p95':>9} {'hardQ.med':>10}  [um]")
    sweep_out = {}
    aq = np.abs(data["qop"]); hard = aq >= np.percentile(aq, 75)
    mean_steps = np.mean([int(round((data["z1"][i]-data["z0"][i]) / 1.0)) for i in range(N_TRACKS)])
    for dz in DZ_SWEEP:
        r = _run_all(field, data, dz, buggy=True, dtype=np.float32) - truth
        pos = _pos_um(r)
        nsteps = np.mean([int(round(abs(data["z1"][i]-data["z0"][i]) / dz)) for i in range(N_TRACKS)])
        evals = 6.0 * nsteps                       # Cash-Karp = 6 field evals/step
        sweep_out[dz] = {"evals_per_track": evals, "pos_med_um": float(np.percentile(pos, 50)),
                         "pos_p95_um": float(np.percentile(pos, 95)),
                         "hardQ_med_um": float(np.percentile(pos[hard], 50))}
        print(f"  {dz:>7.0f} {evals:>10.0f} {np.percentile(pos,50):>9.2f} "
              f"{np.percentile(pos,95):>9.2f} {np.percentile(pos[hard],50):>10.2f}")

    # --- 5. fp32 noise floor (dz=50, buggy): fp32 vs fp64 ----------------
    nfp = min(40, N_TRACKS)
    d50_64 = np.array([rk_allen_cashkarp(field, data["z0"][i], data["z1"][i], data["s0"][i],
                                         data["qop"][i], dz=50.0, buggy=True, dtype=np.float64)
                       for i in range(nfp)])
    d50_32 = np.array([rk_allen_cashkarp(field, data["z0"][i], data["z1"][i], data["s0"][i],
                                         data["qop"][i], dz=50.0, buggy=True, dtype=np.float32)
                       for i in range(nfp)])
    floor = scorer.fp32_floor_um(d50_64, d50_32)
    print(f"\nfp32 noise floor (fp32 vs fp64 incumbent, {nfp} tracks): "
          f"median {floor['median']:.3f} um, p95 {floor['p95']:.3f} um")

    # --- 6. Jacobian-gate demo (incumbent vs truth, fp64 to avoid fp32 noise)
    j = int(np.argmax(aq))
    z0, z1, s0, qop = data["z0"][j], data["z1"][j], data["s0"][j], data["qop"][j]
    Jt = truth_jacobian(field, z0, z1, s0, qop)
    eps = (1e-3, 1e-3, 1e-6, 1e-6); Ji = np.zeros((4, 4))
    for c in range(4):
        sp = np.asarray(s0, float).copy(); sm = sp.copy(); sp[c] += eps[c]; sm[c] -= eps[c]
        Ji[:, c] = (rk_allen_cashkarp(field, z0, z1, sp, qop, dz=DZ_DEPLOY, buggy=True, dtype=np.float64)
                    - rk_allen_cashkarp(field, z0, z1, sm, qop, dz=DZ_DEPLOY, buggy=True, dtype=np.float64)) / (2 * eps[c])
    fr = frob_rel(Ji, Jt)
    print(f"\nJacobian gate (incumbent vs truth, hardest track p={C_LIGHT/abs(qop):.1f} GeV): "
          f"frob_rel = {fr:.2e}  ({'PASS' if fr < 0.05 else 'FAIL'} < 0.05)")

    report = {
        "field": field.path, "n_tracks": N_TRACKS, "dz_deploy_mm": DZ_DEPLOY,
        "incumbent_vs_truth_pos_um": {"median": med, "ci95": [lo, hi]},
        "bug_cost_um": {"buggy_median": buggy_med, "corrected_median": cor_med,
                        "buggy_p95": buggy_p95, "corrected_p95": cor_p95},
        "step_size_sweep": {str(k): v for k, v in sweep_out.items()},
        "fp32_floor_um": floor, "jacobian_frob_rel": fr,
        "stratified_pos_um": {label: st["pos"] for label, _m, st in rows},
        "total_seconds": time.time() - t0,
    }
    with open(os.path.join(HERE, "phase0_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> phase0_report.json   (total {report['total_seconds']:.1f}s)")
    print("PHASE 0 (rebuilt) foundation: OK")


if __name__ == "__main__":
    main()
