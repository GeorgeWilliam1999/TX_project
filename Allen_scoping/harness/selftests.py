#!/usr/bin/env python3
"""
selftests.py -- the kappa-guard.  Fail-loud startup checks.

These exist because a x1000-too-weak field (the kappa bug) once survived three
model generations and dozens of internal gates -- every instrument was built
from the same corpus.  The harness refuses to run unless it reproduces the
externally-known field anchors AND the *dynamic* pT kick from a real
integration (which a x1000 error could not fake).

Anchors (LOCKED): peak By = -1.048 T @ z=4700; int By.dl (z 2665->7826) = -3.733 T.m;
pT kick = 0.299792458 * |int B.dl| = 1.12 GeV.
"""
from __future__ import annotations
import numpy as np

from fieldmap import FieldMap, C_LIGHT, C_LIGHT_ALLEN
from integrators import truth_endpoint, convergence_study


class GuardFailure(RuntimeError):
    pass


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise GuardFailure(f"{name}: {detail}")
    return ok


def run(field: FieldMap | None = None, verbose=True):
    field = field or FieldMap()
    if verbose:
        print(f"kappa-guard on {field.path}")
        print(f"  grid {field.Nx}x{field.Ny}x{field.Nz}, spacing {field.spacing} mm, min {field.min}")

    # 0. qop convention: GeV form == Allen MeV form (ExtrapolateStates.cu:43)
    p_gev = 7.5
    q_over_p_mev = 1.0 / (p_gev * 1000.0)
    _check("qop convention (GeV form == Allen c_light*q/p[MeV])",
           abs(C_LIGHT * 1.0 / p_gev - C_LIGHT_ALLEN * q_over_p_mev) < 1e-15,
           f"{C_LIGHT * 1.0 / p_gev:.10f} == {C_LIGHT_ALLEN * q_over_p_mev:.10f}")

    # 1. the speed-of-light constant feeding qop
    _check("c_light", abs(C_LIGHT - 0.299792458) < 1e-12, f"{C_LIGHT}")

    # 2. raw MagDown: peak By on axis is negative, ~ -1.048 T
    zc = field.axis_z(); Bya = field.By_on_axis()
    ipk = int(np.argmin(Bya)); peakT = Bya[ipk] * 1000.0; zpk = zc[ipk]
    _check("By sign (raw MagDown By<0)", peakT < 0, f"peak {peakT:.3f} T @ z={zpk:.0f} mm")
    _check("By peak magnitude ~ -1.048 T", abs(peakT - (-1.048)) < 0.05, f"{peakT:.3f} T")

    # 3. integrated bending power over UT->T on axis ~ -3.733 T.m
    m = (zc >= 2665) & (zc <= 7826)
    intBydl = np.trapezoid(Bya[m], zc[m])
    _check("int By.dl (UT->T) ~ -3.733 T.m", abs(intBydl - (-3.733)) < 0.05, f"{intBydl:.4f} T.m")

    # 4. static pT kick from the integral
    pT_static = C_LIGHT * abs(intBydl)
    _check("static pT kick ~ 1.12 GeV", abs(pT_static - 1.12) < 0.03, f"{pT_static:.4f} GeV")

    # 5. DYNAMIC kappa chain: integrate p=10, q=+1 from (2665->7826), read tx_final.
    p = 10.0
    qop = C_LIGHT * 1.0 / p
    s_end = truth_endpoint(field, 2665.0, 7826.0, (0.0, 0.0, 0.0, 0.0), qop)
    pT_dyn = p * abs(s_end[2])
    _check("dynamic pT kick in [0.8,1.5] GeV (kappa chain)", 0.8 < pT_dyn < 1.5,
           f"{pT_dyn:.4f} GeV  (tx_final={s_end[2]:.5f}, dx={s_end[0]:.1f} mm)")

    # 6. truth is converged: tightening tolerance moves the endpoint < 1 um
    rows = convergence_study(field, 2665.0, 7826.0, (0.0, 0.0, 0.05, 0.02), qop)
    last_delta_um = rows[-1][2] * 1000.0
    if verbose:
        print("  convergence (rtol, x_end[mm], |dx_vs_prev|[mm]):")
        for t, xe, d in rows:
            print(f"      {t:.0e}   {xe:12.6f}   {d:.3e}")
    _check("truth converged (last |dx| < 1 um)", last_delta_um < 1.0, f"{last_delta_um:.4e} um")

    print("kappa-guard: ALL CHECKS PASSED")
    return True


if __name__ == "__main__":
    run()
