#!/usr/bin/env python3
"""
truth.py -- the reference (truth) integrator and the RK baseline.

Two integrators over the SAME field + RHS, deliberately different schemes so a
shared-implementation bug cannot hide (the kappa-bug lesson):

  * truth_endpoint()      adaptive DOP853 (fp64, tight tol) -- the truth of record.
  * cashkarp_endpoint()   fixed-step Cash-Karp(5)           -- the RK baseline /
                          cross-check (the incumbent in the general-step regime).

State convention: s = (x, y, tx, ty);  qop = c*q/p is a constant parameter
(no dE/dx in the long-track chain, so dqop/dz = 0).  The field is in Gaudi units,
so the slope RHS is qop * N * (...B_stored...) -- this carries kappa = 1e-3*qop.

ODE mirrors Allen ExtrapolatorCommon.cuh:46-54.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from fieldmap import FieldMap, C_LIGHT


def rhs(z, s, qop, field: FieldMap):
    """d/dz of (x,y,tx,ty).  Allen Lorentz RHS; field in Gaudi units."""
    x, y, tx, ty = s
    bx, by, bz = field.at(x, y, z)
    norm = np.sqrt(1.0 + tx * tx + ty * ty)
    ax = norm * (ty * (tx * bx + bz) - (1.0 + tx * tx) * by)
    ay = norm * (-tx * (ty * by + bz) + (1.0 + ty * ty) * bx)
    return np.array([tx, ty, qop * ax, qop * ay])


# ----------------------------------------------------------- truth (DOP853)
def truth_endpoint(field, z0, z1, s0, qop, rtol=1e-11, atol=1e-12):
    """Adaptive fp64 endpoint state at z1.  s0 = (x,y,tx,ty)."""
    sol = solve_ivp(rhs, (z0, z1), np.asarray(s0, float), method="DOP853",
                    args=(qop, field), rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"DOP853 failed: {sol.message}")
    return sol.y[:, -1]


# --------------------------------------------------- Cash-Karp(5) fixed step
_CK_C = np.array([0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8])
_CK_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [3 / 10, -9 / 10, 6 / 5],
    [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
    [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
]
_CK_B5 = np.array([37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771])


def cashkarp_endpoint(field, z0, z1, s0, qop, n_steps=512, dtype=np.float64):
    """Fixed-step Cash-Karp(5) endpoint -- the clean RK reference.

    n_steps ~ Allen's step budget; dtype=np.float32 exposes the fp32 floor."""
    s = np.asarray(s0, dtype=dtype).copy()
    h = dtype((z1 - z0) / n_steps)
    z = dtype(z0)
    qop = dtype(qop)
    for _ in range(n_steps):
        k = []
        for i in range(6):
            si = s.copy()
            for j in range(i):
                si = si + h * _CK_A[i][j] * k[j]
            zi = z + _CK_C[i] * h
            bx, by, bz = field.at(float(si[0]), float(si[1]), float(zi))
            tx, ty = si[2], si[3]
            norm = np.sqrt(1.0 + tx * tx + ty * ty)
            ax = norm * (ty * (tx * bx + bz) - (1.0 + tx * tx) * by)
            ay = norm * (-tx * (ty * by + bz) + (1.0 + ty * ty) * bx)
            k.append(np.array([tx, ty, qop * ax, qop * ay], dtype=dtype))
        s = s + h * sum(_CK_B5[i] * k[i] for i in range(6))
        z = z + h
    return s


# --------------------------------------------------------- convergence study
def convergence_study(field, z0, z1, s0, qop, tols=(1e-6, 1e-8, 1e-10, 1e-12)):
    """Integrate at tightening tolerance; report endpoint x and successive |dx|.

    Returns list of (rtol, x_end_mm, delta_vs_previous_mm)."""
    rows, prev = [], None
    for t in tols:
        xe = truth_endpoint(field, z0, z1, s0, qop, rtol=t, atol=t * 1e-1)[0]
        d = abs(xe - prev) if prev is not None else np.nan
        rows.append((t, xe, d)); prev = xe
    return rows


# ----------------------------------------------------------------- Jacobian
def truth_jacobian(field, z0, z1, s0, qop, eps=(1e-3, 1e-3, 1e-6, 1e-6)):
    """4x4 d(endpoint)/d(start) by central finite differences on the truth."""
    s0 = np.asarray(s0, float)
    J = np.zeros((4, 4))
    for j in range(4):
        sp = s0.copy(); sm = s0.copy()
        sp[j] += eps[j]; sm[j] -= eps[j]
        ep = truth_endpoint(field, z0, z1, sp, qop)
        em = truth_endpoint(field, z0, z1, sm, qop)
        J[:, j] = (ep - em) / (2 * eps[j])
    return J


def frob_rel(J_cand, J_truth):
    """Relative Frobenius distance -- the A4 Jacobian gate metric (< 0.05)."""
    return np.linalg.norm(J_cand - J_truth) / np.linalg.norm(J_truth)
