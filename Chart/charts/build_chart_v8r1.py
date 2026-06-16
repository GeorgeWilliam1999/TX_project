#!/usr/bin/env python3
"""F4b — rebuild the analytic chart tables on the REAL LHCb v8r1 field.

extrapUTT's coefficients are fit to v8r1; the canonical chart tables
(chart_tables.npz) are built from the toy twodip field. For a deployment-relevant
bake-off both must use the SAME field. This rebuilds the even-multipole c_ab(z)
tables on v8r1, reusing the exact F3.1 winner config from chart.py.

kappa0 (qop->curvature Lorentz constant in the corpus unit system) is
field-INDEPENDENT in principle. We recalibrate it on the v8r1 PV-pointing truth
pool as a cross-check; if it matches the toy kappa0 we keep the (more robust)
toy value. The field enters ONLY through the By multipole tables C.

Output: charts/chart_tables_v8r1.npz  (same schema as chart_tables.npz).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent.parent / "gen_3"
PAPER = GEN3 / "paper_p0"
sys.path.insert(0, str(GEN3 / "utils"))
sys.path.insert(0, str(PAPER))

from magnetic_field import C_LIGHT  # noqa: E402  (same unit convention as the corpus)
from field_v8r1 import FieldV8R1   # noqa: E402
from chart import (ORDER, XN_X, XN_Y, SIGMA_W, NG, ZSTEP, ZMIN, ZMAX, NS,  # noqa: E402
                   CLAMP, even_terms)

ZINI, ZFIN = 2665.0, 7826.0


def v8r1_on_axis_F(field, dz=5.0):
    """On-axis F(z)=int By(0,0,z') dz' on the fine grid, v8r1 field."""
    z = np.arange(ZMIN, ZMAX + dz, dz)
    _, By, _ = field(np.zeros_like(z), np.zeros_like(z), z)
    By = np.asarray(By, np.float64)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (By[1:] + By[:-1]) * np.diff(z))])
    return z, F


def _chart_I1geo(X7, zpl, terms, C, xnx, xny, ns, clamp=True):
    """The chart's own chord integral I1=int By_multipole ds and geo factor.

    Mirrors chart.chart_predict exactly so kappa0 is calibrated against the same
    quadrature the deployment baseline uses (not the on-axis dF, which differs by
    the weighted-fit center offset)."""
    x0, y0, tx0, ty0 = X7[:, 0], X7[:, 1], X7[:, 2], X7[:, 3]
    z0s, dzs = X7[:, 5], X7[:, 6]
    n = len(X7)
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0., 1., ns)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xp = x0[:, None] + tx0[:, None] * (zp - z0s[:, None])
    yp = y0[:, None] + ty0[:, None] * (zp - z0s[:, None])
    un, vn = xp / xnx, yp / xny
    if clamp:
        un = np.clip(un, -1., 1.); vn = np.clip(vn, -1., 1.)
    By = np.zeros((n, ns))
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, ns)
        By += cz * (un**a) * (vn**b)
    dzc = (dzs / (ns - 1))[:, None]
    I1 = np.sum(0.5 * (By[:, 1:] + By[:, :-1]) * dzc, axis=1)
    return I1, geo


def calibrate_kappa(zpl, terms, C):
    """Self-consistent kappa0: dtx_true ~ -kappa0*qop*geo*I1_chart on near-axis
    high-p tracks (where the order-1 chart is near-exact, O(kappa^2) negligible)."""
    d = np.load(PAPER / "v8r1_plane_truth.npz")
    X, Y = d["X_plane"], d["Y_true"]
    n_all = len(X)
    X7 = np.concatenate([X, np.full((n_all, 1), ZINI), np.full((n_all, 1), ZFIN - ZINI)], axis=1)
    m = (np.abs(X[:, 4]) < 0.05) & (np.abs(X[:, 2]) < 0.15) & (np.abs(X[:, 3]) < 0.15)
    n = int(m.sum())
    I1, geo = _chart_I1geo(X7[m], zpl, terms, C, XN_X, XN_Y, NS)
    u = -(X[m, 4].astype(np.float64)) * geo * I1
    dtx = (Y[m, 2] - X[m, 2]).astype(np.float64)
    k0 = float(np.dot(u, dtx) / np.dot(u, u)) if n > 5 else float("nan")
    r2 = 1.0 - np.var(dtx - k0 * u) / np.var(dtx) if n > 5 else float("nan")
    return k0, r2, n


def build(out=None):
    field = FieldV8R1()
    print("v8r1:", field.info())

    # even-multipole fit of By(x,y,z) on v8r1, identical config to chart.build_chart
    axx = np.linspace(-XN_X, XN_X, NG)
    axy = np.linspace(-XN_Y, XN_Y, NG)
    XX, YY = np.meshgrid(axx, axy, indexing="ij")
    xf, yf = XX.ravel(), YY.ravel()
    w = np.exp(-(xf**2 + yf**2) / (2 * SIGMA_W**2))
    sw = np.sqrt(w)
    terms = even_terms(ORDER)
    A = np.stack([(xf / XN_X)**a * (yf / XN_Y)**b for (a, b) in terms], axis=1)
    Apinv = np.linalg.pinv(sw[:, None] * A)
    zpl = np.arange(ZMIN, ZMAX + ZSTEP, ZSTEP)
    C = np.empty((len(zpl), len(terms)))
    for i, z in enumerate(zpl):
        _, By, _ = field(xf, yf, np.full_like(xf, z))
        C[i] = Apinv @ (sw * np.asarray(By, np.float64))

    # kappa0 calibrated self-consistently against the chart's OWN chord integral
    k0_toy = float(np.load(HERE / "field_integrals.npz")["kappa0"])
    kappa0, r2, n = calibrate_kappa(zpl, terms, C)
    print(f"kappa0 toy (twodip units)  = {k0_toy:.8e}")
    print(f"kappa0 v8r1 (self-consist) = {kappa0:.8e}  (R^2={r2:.6f}, n={n})")

    out = out or (HERE / "chart_tables_v8r1.npz")
    np.savez(out, z=zpl, C=C, terms=np.array(terms), order=ORDER,
             xn_x=XN_X, xn_y=XN_Y, clamp=CLAMP, ns=NS, kappa0=kappa0,
             c_light=C_LIGHT, source="field.v8r1.down.bin", kappa0_toy=k0_toy)
    print(f"built v8r1 chart: {len(terms)} terms, z[{zpl[0]:.0f},{zpl[-1]:.0f}]@{ZSTEP}mm, "
          f"{C.size*4/1024:.1f} kB -> {out}")
    return out


if __name__ == "__main__":
    build()
