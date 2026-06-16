#!/usr/bin/env python3
"""
fieldmap.py -- Allen magnetic-field map loader + trilinear lookup.

Reads the Allen `magfield.bin` (READ-ONLY input) and exposes the same trilinear
interpolation Allen uses on the GPU (MagneticField.cuh:38-78).  Scalar `.at()`
feeds the adaptive truth integrator; vectorised `.at_batch()` feeds bulk
data-generation.

Units / conventions (LOCKED -- see the Notion scoping to-do, do not re-derive):
  * field = LHCb FieldMap v8r1 down, raw MagDown By < 0, no sign flips.
  * stored values are in Gaudi units (tesla = 1e-3): a 1 T field is stored as
    1e-3.  The ODE multiplies qop * B_stored, which embeds kappa = 1e-3 * qop
    with B in Tesla.  Therefore DO NOT rescale the stored field here.
  * qop fed to the ODE is c*q/p = 0.299792458 * q/p[1/GeV]  (Allen State.qop).
"""
from __future__ import annotations
import struct, array
import numpy as np

C_LIGHT = 0.299792458          # State.qop = C_LIGHT * q/p[1/GeV]
DEFAULT_FIELD = "/data/bfys/gscriven/TE_stack/Allen/input/detector_configuration/magfield.bin"


class FieldMap:
    """LHCb dipole field map with Allen-faithful trilinear interpolation."""

    def __init__(self, path: str = DEFAULT_FIELD):
        raw = open(path, "rb").read()
        self.invD = np.array(struct.unpack_from("<3f", raw, 0), dtype=np.float64)
        self.N = np.array(struct.unpack_from("<3i", raw, 16), dtype=np.int64)
        self.min = np.array(struct.unpack_from("<3f", raw, 32), dtype=np.float64)
        Nx, Ny, Nz = (int(v) for v in self.N)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.spacing = 1.0 / self.invD
        flat = array.array("f")
        flat.frombytes(raw[48:48 + Nx * Ny * Nz * 4 * 4])
        # [iz, iy, ix, (Bx,By,Bz,pad)]
        self.B = np.array(flat, dtype=np.float64).reshape(Nz, Ny, Nx, 4)
        self.path = path

    # ------------------------------------------------------------------ scalar
    def at(self, x: float, y: float, z: float):
        """Trilinear (Bx,By,Bz) in Gaudi units; (0,0,0) outside the grid."""
        fx = (x - self.min[0]) * self.invD[0]
        fy = (y - self.min[1]) * self.invD[1]
        fz = (z - self.min[2]) * self.invD[2]
        ix, iy, iz = int(np.floor(fx)), int(np.floor(fy)), int(np.floor(fz))
        if ix < 0 or iy < 0 or iz < 0 or ix >= self.Nx - 1 or iy >= self.Ny - 1 or iz >= self.Nz - 1:
            return 0.0, 0.0, 0.0
        hx, hy, hz = fx - ix, fy - iy, fz - iz
        B = self.B
        c000 = (1 - hx) * (1 - hy); c100 = hx * (1 - hy)
        c010 = (1 - hx) * hy;       c110 = hx * hy
        out = []
        for k in range(3):
            v = ((1 - hz) * (c000 * B[iz, iy, ix, k]     + c100 * B[iz, iy, ix + 1, k]
                            + c010 * B[iz, iy + 1, ix, k] + c110 * B[iz, iy + 1, ix + 1, k])
                  + hz * (c000 * B[iz + 1, iy, ix, k]     + c100 * B[iz + 1, iy, ix + 1, k]
                          + c010 * B[iz + 1, iy + 1, ix, k] + c110 * B[iz + 1, iy + 1, ix + 1, k]))
            out.append(v)
        return out[0], out[1], out[2]

    # -------------------------------------------------------------- vectorised
    def at_batch(self, x, y, z):
        """Vectorised trilinear lookup. x,y,z are arrays of equal shape.

        Returns (Bx,By,Bz) arrays; points outside the grid get 0."""
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        fx = (x - self.min[0]) * self.invD[0]
        fy = (y - self.min[1]) * self.invD[1]
        fz = (z - self.min[2]) * self.invD[2]
        ix = np.floor(fx).astype(np.int64); iy = np.floor(fy).astype(np.int64)
        iz = np.floor(fz).astype(np.int64)
        inside = ((ix >= 0) & (iy >= 0) & (iz >= 0)
                  & (ix < self.Nx - 1) & (iy < self.Ny - 1) & (iz < self.Nz - 1))
        ixc = np.clip(ix, 0, self.Nx - 2); iyc = np.clip(iy, 0, self.Ny - 2)
        izc = np.clip(iz, 0, self.Nz - 2)
        hx = fx - ixc; hy = fy - iyc; hz = fz - izc
        B = self.B
        def gather(dz, dy_, dx):
            return B[izc + dz, iyc + dy_, ixc + dx, :3]
        w = lambda a, b, c: ((a) * (b) * (c))[:, None]
        val = (w(1 - hz, 1 - hy, 1 - hx) * gather(0, 0, 0)
               + w(1 - hz, 1 - hy, hx) * gather(0, 0, 1)
               + w(1 - hz, hy, 1 - hx) * gather(0, 1, 0)
               + w(1 - hz, hy, hx) * gather(0, 1, 1)
               + w(hz, 1 - hy, 1 - hx) * gather(1, 0, 0)
               + w(hz, 1 - hy, hx) * gather(1, 0, 1)
               + w(hz, hy, 1 - hx) * gather(1, 1, 0)
               + w(hz, hy, hx) * gather(1, 1, 1))
        val = val * inside[:, None]
        return val[:, 0], val[:, 1], val[:, 2]

    # ----------------------------------------------------------------- helpers
    def axis_z(self):
        return self.min[2] + np.arange(self.Nz) * self.spacing[2]

    def By_on_axis(self):
        ix0 = int(round((0 - self.min[0]) * self.invD[0]))
        iy0 = int(round((0 - self.min[1]) * self.invD[1]))
        return self.B[:, iy0, ix0, 1]
