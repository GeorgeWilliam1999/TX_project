#!/usr/bin/env python3
"""F4b — v8r1 + PV-pointing bake-off: chart (rebuilt on v8r1) vs extrapUTT vs straight.

Common field: real LHCb v8r1 (down). Common population: PV-pointing plane states
(v8r1_plane_truth.npz). Both the chart tables and extrapUTT's coefficients are
fit to v8r1, so this is the deployment-relevant, apples-to-apples comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent           # experiments/flattening/results
FLAT = HERE.parent
GEN3 = FLAT.parent / "gen_3"
PAPER = GEN3 / "paper_p0"
sys.path.insert(0, str(FLAT / "charts"))
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(GEN3 / "utils"))

from chart import chart_predict, load_chart      # noqa: E402
from extraputt_py import read_params_UTT, extrapUTT, _DEFAULT_TAB  # noqa: E402

ZINI, ZFIN = 2665.0, 7826.0
DZ = ZFIN - ZINI


def metrics(pred, truth):
    dx = np.abs(pred[:, 0] - truth[:, 0]) * 1e3   # um
    dy = np.abs(pred[:, 1] - truth[:, 1]) * 1e3
    dtx = np.abs(pred[:, 2] - truth[:, 2]) * 1e6  # urad
    return dict(med=np.median(dx), p95=np.quantile(dx, 0.95), p99=np.quantile(dx, 0.99),
                mdy=np.median(dy), mdtx=np.median(dtx))


def row(tag, m):
    print(f"  {tag:<26} med dx={m['med']:>9.1f}  p95={m['p95']:>9.1f}  p99={m['p99']:>9.1f}  "
          f"dy={m['mdy']:>8.1f}  dtx={m['mdtx']:>8.1f} urad")


def main():
    d = np.load(PAPER / "v8r1_plane_truth.npz")
    X5, Y = d["X_plane"], d["Y_true"]   # (x,y,tx,ty,qop) @ZINI ; truth @ZFIN on v8r1
    n = len(X5)
    qop = X5[:, 4]
    print(f"v8r1 PV-pointing pool: {n} tracks, z {ZINI}->{ZFIN}; "
          f"p in [{0.299792458/np.abs(qop).max():.1f}, {0.299792458/np.abs(qop).min():.0f}] GeV")
    print(f"  PV-pointing check: median |x - tx*zi| = {np.median(np.abs(X5[:,0]-X5[:,2]*ZINI)):.1f} mm\n")

    # straight line
    sl = np.stack([X5[:, 0] + X5[:, 2] * DZ, X5[:, 1] + X5[:, 3] * DZ, X5[:, 2], X5[:, 3]], axis=1)
    row("straight_line", metrics(sl, Y))

    # extrapUTT (native v8r1, pol -1)
    p = read_params_UTT(_DEFAULT_TAB)
    eu = extrapUTT(p, X5, polarity=-1, scale_qop=True)
    row("extrapUTT (pol -1)", metrics(eu, Y))

    # chart rebuilt on v8r1
    X7 = np.concatenate([X5, np.full((n, 1), ZINI), np.full((n, 1), DZ)], axis=1)
    ch_v8 = load_chart(FLAT / "charts" / "chart_tables_v8r1.npz")
    cv = chart_predict(X7, chart=ch_v8)
    row("chart (v8r1 tables)", metrics(cv, Y))

    # for reference: the toy-field chart on the same inputs (expected to be off -> field mismatch)
    ch_toy = load_chart(FLAT / "charts" / "chart_tables.npz")
    ct = chart_predict(X7, chart=ch_toy)
    row("chart (toy tables) [ref]", metrics(ct, Y))

    # quartile breakdown for the two field-matched contenders
    print("\n  median dx by |q/p| quartile (low p = high |qop| = Q4):")
    q = np.abs(qop)
    edges = np.quantile(q, [0.25, 0.5, 0.75])
    b = np.digitize(q, edges)
    for tag, pred in [("extrapUTT", eu), ("chart v8r1", cv)]:
        dx = np.abs(pred[:, 0] - Y[:, 0]) * 1e3
        bq = "/".join(f"{np.median(dx[b == k]):.1f}" for k in range(4))
        print(f"    {tag:<12} [{bq}] um")


if __name__ == "__main__":
    main()
