#!/usr/bin/env python3
"""
scorer.py -- paired residual scoring for the bake-off.

Every candidate is run on the SAME inputs as the truth, so residuals are paired
(candidate - truth).  We report the full distribution per component, stratified
by momentum, with bootstrap CIs -- never a single aggregate median (that is how
the v1 residual model silently worsened the easy quartile).

Position residuals -> micrometres; slope residuals -> microradians.
"""
from __future__ import annotations
import numpy as np

# state index map
IX, IY, ITX, ITY = 0, 1, 2, 3
POS_TO_UM = 1.0e3      # mm  -> um
SLOPE_TO_URAD = 1.0e6  # rad -> urad  (tx,ty are slopes ~ tan(theta) ~ theta)


def _pct(a, q):
    return float(np.percentile(np.abs(a), q))


def component_stats(resid):
    """resid: (N,4) candidate-truth.  Returns dict of per-component metrics."""
    r = np.asarray(resid)
    out = {}
    for name, idx, scale, unit in (("x", IX, POS_TO_UM, "um"), ("y", IY, POS_TO_UM, "um"),
                                   ("tx", ITX, SLOPE_TO_URAD, "urad"),
                                   ("ty", ITY, SLOPE_TO_URAD, "urad")):
        d = r[:, idx] * scale
        out[name] = {"unit": unit, "median": _pct(d, 50), "p95": _pct(d, 95),
                     "p99": _pct(d, 99), "max": float(np.max(np.abs(d))),
                     "bias": float(np.mean(d))}
    # combined transverse position error (the headline accuracy number, um)
    pos = np.hypot(r[:, IX], r[:, IY]) * POS_TO_UM
    out["pos"] = {"unit": "um", "median": float(np.percentile(pos, 50)),
                  "p95": float(np.percentile(pos, 95)), "p99": float(np.percentile(pos, 99)),
                  "max": float(np.max(pos))}
    return out


def bootstrap_ci(values, stat=np.median, n_boot=1000, ci=95, seed=0):
    """Bootstrap CI for a scalar statistic of |values|."""
    rng = np.random.default_rng(seed)
    a = np.abs(np.asarray(values)); n = len(a)
    boots = np.array([stat(a[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat(a)), float(lo), float(hi)


def stratify_by_momentum(qop, resid, edges_gev=(3, 5, 10, 20, 50, 1e9)):
    """Group paired residuals into momentum bins via p = C_LIGHT/|qop|.

    Returns list of (label, mask, component_stats) including an 'all' row and
    explicit easy/hard quartiles by |qop|."""
    from fieldmap import C_LIGHT
    qop = np.asarray(qop); resid = np.asarray(resid)
    p = C_LIGHT / np.abs(qop)
    rows = [("all", np.ones(len(p), bool))]
    lo = 0.0
    for hi in edges_gev:
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append((f"p[{lo:g},{hi:g}) GeV", m))
        lo = hi
    # hardest (low-p) and easiest (high-p) quartiles by |qop|
    aq = np.abs(qop)
    q75, q25 = np.percentile(aq, 75), np.percentile(aq, 25)
    rows.append(("hard quartile (high |qop|)", aq >= q75))
    rows.append(("easy quartile (low |qop|)", aq <= q25))
    return [(label, m, component_stats(resid[m])) for label, m in rows if m.sum()]


def fp32_floor_um(truth_states, fp32_states):
    """Achievable lower bound: |fp32 endpoint - fp64 endpoint| transverse, um."""
    t = np.asarray(truth_states); f = np.asarray(fp32_states)
    pos = np.hypot(f[:, IX] - t[:, IX], f[:, IY] - t[:, IY]) * POS_TO_UM
    return {"median": float(np.percentile(pos, 50)), "p95": float(np.percentile(pos, 95)),
            "max": float(np.max(pos))}


def print_table(stats_rows):
    """Pretty-print stratified pos-residual rows."""
    print(f"  {'stratum':<32} {'N':>7} {'pos.med':>9} {'pos.p95':>9} {'pos.p99':>9}  [um]")
    for label, mask, st in stats_rows:
        s = st["pos"]
        print(f"  {label:<32} {int(np.sum(mask)):>7} {s['median']:>9.2f} "
              f"{s['p95']:>9.2f} {s['p99']:>9.2f}")
