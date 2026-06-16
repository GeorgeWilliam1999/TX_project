#!/usr/bin/env python3
"""
sampler.py -- input-state distributions for the bake-off.

Phase-0 provides a synthetic *general-step* sampler: random start plane z0 and
step dz anywhere in the field region, realistic LHCb acceptance slopes, and a
log-uniform (1/p) momentum spectrum over both charges.  This is the distribution
that matters for the GENERAL-step regime (the focus), where the incumbent is RK.

NOTE: this is a controllable placeholder.  Phase 1 replaces / augments it with
REAL MC tracks extracted from Allen output (the external-validity set) -- see the
Notion scoping to-do, Phase 1.
"""
from __future__ import annotations
import numpy as np
from fieldmap import C_LIGHT


def sample_general(n, rng=None, p_gev=(3.0, 100.0), z0_mm=(2700.0, 8500.0),
                   dz_mm=(300.0, 1500.0), tx_max=0.30, ty_max=0.25):
    """Return dict with arrays: z0, z1, s0 (n,4)=(x,y,tx,ty), qop (n,), p, q.

    States are in-acceptance: |x0| < tx_max*z0, |y0| < ty_max*z0 (a track from
    near the beamline), with slopes uniform in the acceptance cone."""
    rng = rng or np.random.default_rng(0)
    u = rng.random(n)
    p = p_gev[0] * (p_gev[1] / p_gev[0]) ** u          # log-uniform -> 1/p density
    q = rng.choice([-1.0, 1.0], n)
    qop = C_LIGHT * q / p
    z0 = rng.uniform(z0_mm[0], z0_mm[1], n)
    dz = rng.uniform(dz_mm[0], dz_mm[1], n)
    z1 = z0 + dz
    tx = rng.uniform(-tx_max, tx_max, n)
    ty = rng.uniform(-ty_max, ty_max, n)
    # transverse position consistent with a track pointing back near the origin,
    # plus a few-mm offset; clipped to the acceptance at z0.
    x0 = np.clip(tx * z0 * rng.uniform(0.3, 1.0, n) + rng.normal(0, 3.0, n),
                 -tx_max * z0, tx_max * z0)
    y0 = np.clip(ty * z0 * rng.uniform(0.3, 1.0, n) + rng.normal(0, 3.0, n),
                 -ty_max * z0, ty_max * z0)
    s0 = np.stack([x0, y0, tx, ty], axis=1)
    return {"z0": z0, "z1": z1, "s0": s0, "qop": qop, "p": p, "q": q}
