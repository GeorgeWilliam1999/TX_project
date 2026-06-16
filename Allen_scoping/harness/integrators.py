#!/usr/bin/env python3
"""
integrators.py -- truth + the FAITHFUL deployed baselines.

REBUILT FROM SOURCE (2026-06-16). Every integrator is grounded in the read-only
Allen tree with file:line, so the benchmark compares against what Allen actually
runs -- not an idealised RK.

  * truth_endpoint()       adaptive DOP853 (fp64, tight tol) -- truth of record.
  * rk_allen_cashkarp()    the DEPLOYED incumbent: fixed-step Cash-Karp(6) with
                           Allen's off-by-one stage bug.  This is exactly what
                           extrapolate_states_t runs (dz=100mm, n_steps=100).
                             ExtrapolateStates.cu:46-49 (loop + call)
                             RungeKuttaExtrapolator.cuh:25-45 (propagate; bug L32)
                             ButcherTableau.cuh:74-107 (Cash-Karp tableau)
  * set buggy=False        -> the CORRECTED Cash-Karp, to MEASURE the bug cost.
  * rk_nystrom_fast()      the other production extrapolator (1 field eval at the
                           midpoint, used by the ttrack/downstream chain)
                             RungeKuttaExtrapolator.cuh:132-150, 230-247.

ODE (ExtrapolatorCommon.cuh:46-54), c_light (line 16). State (x,y,z,tx,ty); qop
is a constant parameter (no dE/dx in this chain).
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from fieldmap import FieldMap

# Cash-Karp tableau, ButcherTableau.cuh:74-107  (a(i,j) = a[i*(i-1)/2 + j])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [3 / 10, -9 / 10, 6 / 5],
    [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
    [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
]
_B = [37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771]
_BSTAR = [2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4]
_NYS_C = [0.0, 0.5, 0.5, 1.0]


# ---------------------------------------------------- RHS (ExtrapolatorCommon.cuh:46-54)
def _slopes(tx, ty, bx, by, bz):
    norm = np.sqrt(1.0 + tx * tx + ty * ty)
    ax = norm * (ty * (tx * bx + bz) - (1.0 + tx * tx) * by)
    ay = norm * (-tx * (ty * by + bz) + (1.0 + ty * ty) * bx)
    return ax, ay


def _rhs4(z, s, qop, field):
    """d/dz of (x,y,tx,ty), z the independent variable -- for the truth solver."""
    x, y, tx, ty = s
    bx, by, bz = field.at(x, y, z)
    ax, ay = _slopes(tx, ty, bx, by, bz)
    return np.array([tx, ty, qop * ax, qop * ay])


# ----------------------------------------------------------- truth (DOP853)
def truth_endpoint(field, z0, z1, s0, qop, rtol=1e-11, atol=1e-12):
    """Adaptive fp64 endpoint (x,y,tx,ty) at z1.  s0 = (x,y,tx,ty)."""
    sol = solve_ivp(_rhs4, (z0, z1), np.asarray(s0, float), method="DOP853",
                    args=(qop, field), rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"DOP853 failed: {sol.message}")
    return sol.y[:, -1]


# --------------------------------------- faithful Allen Cash-Karp (with the bug)
def _deriv5(s5, qop, field, dtype):
    """Allen State::derivative*1 on (x,y,z,tx,ty): returns (tx,ty,1,qop*ax,qop*ay)."""
    bx, by, bz = field.at(float(s5[0]), float(s5[1]), float(s5[2]))
    tx, ty = s5[3], s5[4]
    ax, ay = _slopes(tx, ty, bx, by, bz)
    return np.array([tx, ty, 1.0, qop * ax, qop * ay], dtype=dtype)


def _ck_step(field, s5, qop, h, buggy, dtype):
    """One Cash-Karp step on (x,y,z,tx,ty).  buggy=True reproduces the
    `for i<stage-1` off-by-one (RungeKuttaExtrapolator.cuh:32)."""
    k = [None] * 6
    for stage in range(6):
        s = s5.copy()
        jmax = stage - 1 if buggy else stage          # <-- the deployed bug vs the fix
        for i in range(jmax):
            s = s + k[i] * dtype(_A[stage][i])
        k[stage] = _deriv5(s, qop, field, dtype) * h
    new = s5.copy()
    for i in range(6):
        new = new + k[i] * dtype(_B[i])
    return new


def rk_allen_cashkarp(field, z0, z1, s0, qop, dz=100.0, buggy=True, dtype=np.float32):
    """Deployed incumbent: fixed dz Cash-Karp from z0 to z1.  s0=(x,y,tx,ty).

    Defaults dz=100 mm, dtype=float32 (extrapolate_states_t config & precision).
    Steps tile [z0,z1] exactly (h ~ dz) so the whole interval is integrated.
    Returns (x,y,tx,ty) at z1."""
    n = max(1, int(round(abs(z1 - z0) / dz)))
    h = dtype((z1 - z0) / n)
    s5 = np.array([s0[0], s0[1], z0, s0[2], s0[3]], dtype=dtype)
    qop = dtype(qop)
    for _ in range(n):
        s5 = _ck_step(field, s5, qop, h, buggy, dtype)
    return np.array([s5[0], s5[1], s5[3], s5[4]], dtype=np.float64)


# --------------------------- Nystrom fast-step (ttrack chain), 1 field eval/step
def _gamma(tx, ty, bx, by, bz):
    norm = np.sqrt(1.0 + tx * tx + ty * ty)
    return (norm * (tx * ty * bx - (1.0 + tx * tx) * by + ty * bz),
            norm * ((1.0 + ty * ty) * bx - tx * ty * by - tx * bz))


def rk_nystrom_fast(field, z0, z1, s0, qop, step=500.0, dtype=np.float32):
    """RungeKuttaNystrom::make_fast_step driver (RungeKuttaExtrapolator.cuh:132-247)."""
    x, y, tx, ty = (dtype(v) for v in s0)
    z = dtype(z0); qop = dtype(qop)
    target = dtype(z1); direction = dtype(np.sign(z1 - z0))
    for _ in range(10000):
        if abs(z - target) < 1.0:
            break
        h = direction * min(dtype(step), abs(target - z))
        bx, by, bz = field.at(float(x + 0.5 * tx * h), float(y + 0.5 * ty * h), float(z + 0.5 * h))
        k = [None] * 4
        gx, gy = _gamma(tx, ty, bx, by, bz); k[0] = (qop * gx, qop * gy)
        for st in range(1, 4):
            tnx = tx + k[st - 1][0] * (h * dtype(_NYS_C[st]))
            tny = ty + k[st - 1][1] * (h * dtype(_NYS_C[st]))
            gx, gy = _gamma(tnx, tny, bx, by, bz); k[st] = (qop * gx, qop * gy)
        dRx = tx * h + (k[0][0] + k[1][0] + k[2][0]) * (h * h / dtype(6.0))
        dRy = ty * h + (k[0][1] + k[1][1] + k[2][1]) * (h * h / dtype(6.0))
        dTx = (k[0][0] + 2 * k[1][0] + 2 * k[2][0] + k[3][0]) * (h / dtype(6.0))
        dTy = (k[0][1] + 2 * k[1][1] + 2 * k[2][1] + k[3][1]) * (h / dtype(6.0))
        x += dRx; y += dRy; z += h; tx += dTx; ty += dTy
    return np.array([x, y, tx, ty], dtype=np.float64)


# --------------------------------------------------------- convergence + Jacobian
def convergence_study(field, z0, z1, s0, qop, tols=(1e-6, 1e-8, 1e-10, 1e-12)):
    rows, prev = [], None
    for t in tols:
        xe = truth_endpoint(field, z0, z1, s0, qop, rtol=t, atol=t * 1e-1)[0]
        d = abs(xe - prev) if prev is not None else np.nan
        rows.append((t, xe, d)); prev = xe
    return rows


def truth_jacobian(field, z0, z1, s0, qop, eps=(1e-3, 1e-3, 1e-6, 1e-6)):
    s0 = np.asarray(s0, float); J = np.zeros((4, 4))
    for j in range(4):
        sp = s0.copy(); sm = s0.copy(); sp[j] += eps[j]; sm[j] -= eps[j]
        J[:, j] = (truth_endpoint(field, z0, z1, sp, qop)
                   - truth_endpoint(field, z0, z1, sm, qop)) / (2 * eps[j])
    return J


def frob_rel(J_cand, J_truth):
    return np.linalg.norm(J_cand - J_truth) / np.linalg.norm(J_truth)
