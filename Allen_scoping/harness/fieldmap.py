#!/usr/bin/env python3
"""
fieldmap.py -- Allen magnetic-field map loader + trilinear lookup.

REBUILT FROM SOURCE (2026-06-16). Grounded directly in the authoritative,
read-only Allen tree -- no reuse of prior/archived scripts:

  * binary format + B de-interleave:
      TE_stack/Allen/integration/non_event_data/src/MagneticField.cpp:54-87
      header = 12 floats (48 B): invD[0:3] pad | N[0:3](int) pad | min[0:3] pad;
      then B as interleaved (Bx,By,Bz,pad) stride 4, point index Nx*(Ny*iz+iy)+ix.
  * trilinear interpolation (reproduced exactly):
      Allen/device/event_model/common/include/MagneticField.cuh:38-89
      coords cast to float, (int) truncation, 0 outside grid.

Units / conventions (LOCKED, now verified against source):
  * field = LHCb FieldMap v8r1 down, raw MagDown By < 0, stored in Gaudi units
    (tesla = 1e-3); the ODE multiplies qop * B_stored, embedding kappa = 1e-3*qop
    with B in Tesla. DO NOT rescale the stored field.
  * qop fed to the ODE = c_light * q/p.  Allen uses c_light = 299.792458 with
    q/p in [1/MeV] (ExtrapolateStates.cu:43, ExtrapolatorCommon.cuh:16).  That is
    numerically identical to C_LIGHT = 0.299792458 with p in [GeV] -- the form
    used here.  selftests.py asserts this equivalence.
"""
from __future__ import annotations
import struct, array
import numpy as np

# qop = C_LIGHT * q / p[GeV]   ==   c_light_allen * q / p[MeV]
C_LIGHT = 0.299792458
C_LIGHT_ALLEN = 2.99792458e8 * 1000.0 / 1.0e9          # = 299.792458 (source value)
DEFAULT_FIELD = "/data/bfys/gscriven/TE_stack/Allen/input/detector_configuration/magfield.bin"


def qop_from_p(p_gev, q=1.0):
    """RK State.qop for a track of momentum p_gev and charge q."""
    return C_LIGHT * q / p_gev


class FieldMap:
    """LHCb dipole field map with Allen-faithful trilinear interpolation."""

    def __init__(self, path: str = DEFAULT_FIELD):
        raw = open(path, "rb").read()
        # header: invD (3f) | pad | N (3i) | pad | min (3f) | pad   -> B at byte 48
        self.invD = np.array(struct.unpack_from("<3f", raw, 0), dtype=np.float64)
        self.N = np.array(struct.unpack_from("<3i", raw, 16), dtype=np.int64)
        self.min = np.array(struct.unpack_from("<3f", raw, 32), dtype=np.float64)
        Nx, Ny, Nz = (int(v) for v in self.N)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.spacing = 1.0 / self.invD
        flat = array.array("f")
        flat.frombytes(raw[48:48 + Nx * Ny * Nz * 4 * 4])
        # point index = Nx*(Ny*iz+iy)+ix, 4 floats/point -> [iz,iy,ix,(Bx,By,Bz,pad)]
        self.B = np.array(flat, dtype=np.float64).reshape(Nz, Ny, Nx, 4)
        self.path = path

    # ---- scalar lookup (MagneticField.cuh:38-78), fp64 for the truth integrator
    def at(self, x: float, y: float, z: float):
        fx = (x - self.min[0]) * self.invD[0]
        fy = (y - self.min[1]) * self.invD[1]
        fz = (z - self.min[2]) * self.invD[2]
        ix, iy, iz = int(fx), int(fy), int(fz)           # Allen uses (int) truncation
        if ix < 0 or iy < 0 or iz < 0 or ix >= self.Nx - 1 or iy >= self.Ny - 1 or iz >= self.Nz - 1:
            return 0.0, 0.0, 0.0
        h1x, h1y, h1z = fx - ix, fy - iy, fz - iz
        h0x, h0y, h0z = 1.0 - h1x, 1.0 - h1y, 1.0 - h1z
        h00 = h0x * h0y; h01 = h0x * h1y; h10 = h1x * h0y; h11 = h1x * h1y
        B = self.B
        out = []
        for k in range(3):
            v = (h0z * (h00 * B[iz, iy, ix, k]     + h10 * B[iz, iy, ix + 1, k]
                        + h01 * B[iz, iy + 1, ix, k] + h11 * B[iz, iy + 1, ix + 1, k])
                 + h1z * (h00 * B[iz + 1, iy, ix, k]     + h10 * B[iz + 1, iy, ix + 1, k]
                          + h01 * B[iz + 1, iy + 1, ix, k] + h11 * B[iz + 1, iy + 1, ix + 1, k]))
            out.append(v)
        return out[0], out[1], out[2]

    # ---- vectorised lookup for bulk data-gen (same math, batched)
    def at_batch(self, x, y, z):
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); z = np.asarray(z, np.float64)
        fx = (x - self.min[0]) * self.invD[0]
        fy = (y - self.min[1]) * self.invD[1]
        fz = (z - self.min[2]) * self.invD[2]
        ix = fx.astype(np.int64); iy = fy.astype(np.int64); iz = fz.astype(np.int64)
        inside = ((ix >= 0) & (iy >= 0) & (iz >= 0)
                  & (ix < self.Nx - 1) & (iy < self.Ny - 1) & (iz < self.Nz - 1))
        ixc = np.clip(ix, 0, self.Nx - 2); iyc = np.clip(iy, 0, self.Ny - 2); izc = np.clip(iz, 0, self.Nz - 2)
        h1x = fx - ixc; h1y = fy - iyc; h1z = fz - izc
        h0x = 1 - h1x; h0y = 1 - h1y; h0z = 1 - h1z
        B = self.B
        g = lambda dz, dy_, dx: B[izc + dz, iyc + dy_, ixc + dx, :3]
        w = lambda a, b, c: (a * b * c)[:, None]
        val = (w(h0z, h0y, h0x) * g(0, 0, 0) + w(h0z, h0y, h1x) * g(0, 0, 1)
               + w(h0z, h1y, h0x) * g(0, 1, 0) + w(h0z, h1y, h1x) * g(0, 1, 1)
               + w(h1z, h0y, h0x) * g(1, 0, 0) + w(h1z, h0y, h1x) * g(1, 0, 1)
               + w(h1z, h1y, h0x) * g(1, 1, 0) + w(h1z, h1y, h1x) * g(1, 1, 1))
        val = val * inside[:, None]
        return val[:, 0], val[:, 1], val[:, 2]

    def axis_z(self):
        return self.min[2] + np.arange(self.Nz) * self.spacing[2]

    def By_on_axis(self):
        ix0 = int(round((0 - self.min[0]) * self.invD[0]))
        iy0 = int(round((0 - self.min[1]) * self.invD[1]))
        return self.B[:, iy0, ix0, 1]
