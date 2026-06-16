#!/usr/bin/env python3
"""F4a — clean Python reimplementation of Allen's extrapUTT (UT->T polynomial).

Replicates, exactly, the production analytic kick map used as the bake-off
incumbent:
  - parser  : KalmanParametrizations::read_params_UTT  (KalmanParametrizations.cuh:223)
  - META    : the 19 header scalars (dev_UTT_META indices, ParKalmanMethods.cuh:300)
  - eval    : extrapUTT + compute_state<DEG0,DEG1>  (ParKalmanMethods.cuh:287, .cuh:80)

This module currently implements ONLY the parser + META container (F4a step 1).
The compute_state / extrapUTT evaluation is added after the scoping checkpoint.

Compile-time constants (device/event_model/kalman/include/ParKalmanDefinitions.cuh):
  nBinXMax=60  nBinYMax=50   DEGx1=7  DEGx2=9   DEGy1=5  DEGy2=7
These must match the .tab header (Nbinx Nbiny ... DEGX1 DEGX2 DEGY1 DEGY2).

qop convention (the suspected root cause of the broken poly_pred CSVs):
  corpus qop = 299.792458 * q/p[1/MeV]  (|qop|~0.2998 at p=1 GeV).
  extrapUTT expects qOp = q/p[1/MeV] so that fq = qOp*polarity*PMIN is O(1)
  (PMIN=1500 MeV).  => qOp_extrapUTT = corpus_qop / 299.792458.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allen compile-time constants (ParKalmanDefinitions.cuh).
NBINXMAX = 60
NBINYMAX = 50
DEGX1, DEGX2 = 7, 9
DEGY1, DEGY2 = 5, 7

# corpus_qop = QOP_C * q/p[1/MeV]; divide it out to feed extrapUTT.
QOP_C = 299.792458


@dataclass
class UTTParams:
    """Loaded extrapUTT parametrization: META header + coefficient grids."""

    # --- META header scalars (dev_UTT_META[0..18]) ---
    ZINI: float
    ZFIN: float
    PMIN: float
    BENDX: float
    BENDX_X2: float
    BENDX_Y2: float
    BENDY_XY: float
    Txmax: float
    Tymax: float
    XFmax: float
    Dtxy: float
    Nbinx: int
    Nbiny: int
    XGridOption: int
    YGridOption: int
    DEGX1: int
    DEGX2: int
    DEGY1: int
    DEGY2: int

    # --- coefficient grids, shape [Nbinx, Nbiny, deg], row-major (ix outer) ---
    x00: np.ndarray
    tx00: np.ndarray
    x10: np.ndarray
    x01: np.ndarray
    tx10: np.ndarray
    tx01: np.ndarray
    y00: np.ndarray
    ty00: np.ndarray
    y10: np.ndarray
    y01: np.ndarray
    ty10: np.ndarray
    ty01: np.ndarray

    @property
    def meta(self) -> np.ndarray:
        """The 19-element dev_UTT_META array, in documented index order."""
        return np.array([
            self.ZINI, self.ZFIN, self.PMIN, self.BENDX, self.BENDX_X2,
            self.BENDX_Y2, self.BENDY_XY, self.Txmax, self.Tymax, self.XFmax,
            self.Dtxy, self.Nbinx, self.Nbiny, self.XGridOption,
            self.YGridOption, self.DEGX1, self.DEGX2, self.DEGY1, self.DEGY2,
        ], dtype=np.float64)


def read_params_UTT(path: str | Path) -> UTTParams:
    """Parse a params_UTT_v0.tab exactly as KalmanParametrizations::read_params_UTT.

    Whitespace-token stream (newlines irrelevant, matching `myfile >> ...`).
    Per-bin block order from Read() (KalmanParametrizations.cuh:162); the
    tx*/ty* slope blocks carry the 1e-3 factor Read() applies.
    """
    toks = Path(path).read_text().split()
    it = iter(toks)
    f = lambda: float(next(it))          # noqa: E731
    i = lambda: int(float(next(it)))     # noqa: E731

    ZINI, ZFIN, PMIN = f(), f(), f()
    BENDX, BENDX_X2, BENDX_Y2, BENDY_XY = f(), f(), f(), f()
    Txmax, Tymax, XFmax, Dtxy = f(), f(), f(), f()
    Nbinx, Nbiny = i(), i()
    XGridOption, YGridOption = i(), i()
    dx1, dx2, dy1, dy2 = i(), i(), i(), i()

    assert (dx1, dx2, dy1, dy2) == (DEGX1, DEGX2, DEGY1, DEGY2), \
        f"DEG header {(dx1, dx2, dy1, dy2)} != compile-time {(DEGX1, DEGX2, DEGY1, DEGY2)}"
    assert (Nbinx, Nbiny) == (NBINXMAX, NBINYMAX), \
        f"bin header {(Nbinx, Nbiny)} != compile-time {(NBINXMAX, NBINYMAX)}"

    def grid(deg: int) -> np.ndarray:
        return np.zeros((Nbinx, Nbiny, deg), dtype=np.float64)

    x00, tx00 = grid(dx2), grid(dx2)
    x10, x01, tx10, tx01 = grid(dx1), grid(dx1), grid(dx1), grid(dx1)
    y00, ty00 = grid(dy2), grid(dy2)
    y10, y01, ty10, ty01 = grid(dy1), grid(dy1), grid(dy1), grid(dy1)

    def block(arr: np.ndarray, ix: int, iy: int, deg: int, scale: float = 1.0) -> None:
        for k in range(deg):
            arr[ix, iy, k] = scale * f()

    for ix in range(Nbinx):
        for iy in range(Nbiny):
            block(x00, ix, iy, dx2)
            block(tx00, ix, iy, dx2, 1e-3)
            block(x10, ix, iy, dx1)
            block(x01, ix, iy, dx1)
            block(tx10, ix, iy, dx1, 1e-3)
            block(tx01, ix, iy, dx1, 1e-3)
            block(y00, ix, iy, dy2)
            block(ty00, ix, iy, dy2, 1e-3)
            block(y10, ix, iy, dy1)
            block(y01, ix, iy, dy1)
            block(ty10, ix, iy, dy1, 1e-3)
            block(ty01, ix, iy, dy1, 1e-3)

    leftover = sum(1 for _ in it)
    if leftover:
        raise ValueError(f"{leftover} unparsed tokens remain after {Nbinx}x{Nbiny} bins")

    return UTTParams(
        ZINI, ZFIN, PMIN, BENDX, BENDX_X2, BENDX_Y2, BENDY_XY, Txmax, Tymax,
        XFmax, Dtxy, Nbinx, Nbiny, XGridOption, YGridOption, dx1, dx2, dy1, dy2,
        x00, tx00, x10, x01, tx10, tx01, y00, ty00, y10, y01, ty10, ty01,
    )


def _compute_state(arr00, arr10, arr01, deg0, deg1,
                   ix, iy, gx, gy, sx, sy, rx, ry, ux, uy, fq, state):
    """Vectorised port of compute_state<DEG0,DEG1> (KalmanParametrizations.cuh:80).

    `state` is the running value (straight-line x/y, or original tx/ty); the
    polynomial correction is ADDED, mirroring the by-reference accumulation.
    All track-indexed inputs are shape [N]; arr* are [Nbinx, Nbiny, deg].
    """
    g_xx, g_yy, g_xy = gx * gx, gy * gy, gx * gy
    nrx, nry = 1 - rx, 1 - ry  # C++ !rx, !ry
    scaler0 = (g_yy - gy) * 0.5 + nry * sx * g_xy
    scaler1 = 1.0 - (g_xx + g_yy) + sx * sy * g_xy
    scaler2 = (g_yy + gy) * 0.5 - ry * sx * g_xy
    scaler3 = (g_xx - gx) * 0.5 + nrx * sy * g_xy
    scaler4 = sx * sy * g_xy
    scaler5 = (g_xx + gx) * 0.5 - rx * sy * g_xy

    def interp(arr):  # 6-point biquadratic -> coef[N, deg]
        return (arr[ix, iy - 1] * scaler0[:, None]
                + arr[ix, iy] * scaler1[:, None]
                + arr[ix, iy + 1] * scaler2[:, None]
                + arr[ix - 1, iy] * scaler3[:, None]
                + arr[ix + sx, iy + sy] * scaler4[:, None]
                + arr[ix + 1, iy] * scaler5[:, None])

    # 00 term: state += sum_deg coef[deg] * fq^(deg+1)
    c00 = interp(arr00)
    pow0 = fq[:, None] ** np.arange(1, deg0 + 1)[None, :]
    state = state + np.sum(c00 * pow0, axis=1)

    # 10 (ux) and 01 (uy) terms: state += u * sum_deg coef[deg] * fq^(deg+1)
    pow1 = fq[:, None] ** np.arange(1, deg1 + 1)[None, :]
    c10 = interp(arr10)
    state = state + ux * np.sum(c10 * pow1, axis=1)
    c01 = interp(arr01)
    state = state + uy * np.sum(c01 * pow1, axis=1)
    return state


def extrapUTT(p: UTTParams, X: np.ndarray, polarity: int, scale_qop: bool = True):
    """Vectorised port of extrapUTT (ParKalmanMethods.cuh:287).

    X[N,5] = (x, y, tx, ty, qop_corpus) at z=ZINI. Returns out[N,4]=(x,y,tx,ty)
    at z=ZFIN. `scale_qop` divides corpus qop by 299.792458 to recover q/p[1/MeV];
    set False to feed corpus qop raw (the suspected broken convention).
    """
    x = X[:, 0].astype(np.float64).copy()
    y = X[:, 1].astype(np.float64).copy()
    tx = X[:, 2].astype(np.float64).copy()
    ty = X[:, 3].astype(np.float64).copy()
    qOp = X[:, 4].astype(np.float64).copy()
    if scale_qop:
        qOp = qOp / QOP_C

    qop = qOp * polarity
    zi, zf = p.ZINI, p.ZFIN
    Nbinx, Nbiny = p.Nbinx, p.Nbiny

    xx = x / (zi * p.Txmax)
    yy = y / (zi * p.Tymax)
    dxf = Nbinx * (xx + 1.0) / 2.0
    ix = dxf.astype(np.int64)          # C++ int cast: truncate toward zero
    dxf = dxf - ix
    dyf = Nbiny * (yy + 1.0) / 2.0
    iy = dyf.astype(np.int64)
    dyf = dyf - iy

    DtxyInv = 1.0 / p.Dtxy
    ziInv = 1.0 / zi
    bendx = p.BENDX + p.BENDX_X2 * (x * ziInv) ** 2 + p.BENDX_Y2 * (y * ziInv) ** 2
    bendy = p.BENDY_XY * (x * ziInv) * (y * ziInv)
    ux = (tx - x * ziInv - bendx * qop) * DtxyInv
    uy = (ty - y * ziInv - bendy * qop) * DtxyInv

    gx = dxf - 0.5
    gy = dyf - 0.5
    mlo, mhi = ix <= 0, ix >= Nbinx - 1
    gx = np.where(mlo, gx - 1.0, gx)
    gx = np.where(mhi, gx + 1.0, gx)
    ix = np.where(mlo, 1, ix)
    ix = np.where(mhi, Nbinx - 2, ix)
    nlo, nhi = iy <= 0, iy >= Nbiny - 1
    gy = np.where(nlo, gy - 1.0, gy)
    gy = np.where(nhi, gy + 1.0, gy)
    iy = np.where(nlo, 1, iy)
    iy = np.where(nhi, Nbiny - 2, iy)

    rx = (gx >= 0).astype(np.int64)
    sx = 2 * rx - 1
    ry = (gy >= 0).astype(np.int64)
    sy = 2 * ry - 1

    x_out = x + tx * (zf - zi)
    y_out = y + ty * (zf - zi)
    fq = qop * p.PMIN

    args = dict(ix=ix, iy=iy, gx=gx, gy=gy, sx=sx, sy=sy, rx=rx, ry=ry,
                ux=ux, uy=uy, fq=fq)
    x_out = _compute_state(p.x00, p.x10, p.x01, p.DEGX2, p.DEGX1, state=x_out, **args)
    tx_out = _compute_state(p.tx00, p.tx10, p.tx01, p.DEGX2, p.DEGX1, state=tx, **args)
    y_out = _compute_state(p.y00, p.y10, p.y01, p.DEGY2, p.DEGY1, state=y_out, **args)
    ty_out = _compute_state(p.ty00, p.ty10, p.ty01, p.DEGY2, p.DEGY1, state=ty, **args)
    return np.stack([x_out, y_out, tx_out, ty_out], axis=1)


_DEFAULT_TAB = (
    "/data/bfys/gscriven/TE_stack/PARAM/ParamFiles/data/"
    "ParametrizedKalmanFit/25v0/params_UTT_v0.tab"
)


if __name__ == "__main__":
    import sys

    tab = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TAB
    p = read_params_UTT(tab)
    print(f"parsed: {tab}")
    print(f"META = {p.meta}")
    print(f"  ZINI={p.ZINI} ZFIN={p.ZFIN} PMIN={p.PMIN} polarity-ready")
    print(f"  BENDX={p.BENDX} X2={p.BENDX_X2} Y2={p.BENDX_Y2} XY={p.BENDY_XY}")
    print(f"  Txmax={p.Txmax} Tymax={p.Tymax} Dtxy={p.Dtxy} (DtxyInv={1/p.Dtxy:g})")
    print(f"  grid {p.Nbinx}x{p.Nbiny}; DEG x1/x2/y1/y2 = "
          f"{p.DEGX1}/{p.DEGX2}/{p.DEGY1}/{p.DEGY2}")
    for nm in ["x00", "tx00", "x10", "x01", "tx10", "tx01",
               "y00", "ty00", "y10", "y01", "ty10", "ty01"]:
        a = getattr(p, nm)
        print(f"  {nm:5s} shape={a.shape}  bin(0,0)[:3]={a[0,0,:3]}")
    # cross-check the very first .tab data tokens against bin (0,0)
    print("\nspot-check bin(0,0):")
    print(f"  x00[0,0]  = {p.x00[0,0]}")
    print(f"  tx00[0,0] = {p.tx00[0,0]}   (raw .tab * 1e-3)")
