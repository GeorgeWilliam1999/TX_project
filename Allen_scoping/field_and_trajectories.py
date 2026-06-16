#!/usr/bin/env python3
"""
Allen track-extrapolation study aids.

Reads the Allen magfield.bin (READ-ONLY) and produces:
  - fig1_By_axis.png       By(0,0,z) dipole profile with detector regions
  - fig2_By_xz_slice.png   By(x,0,z) heat map (the bending plane)
  - fig3_B_components.png   Bx,By,Bz along the axis
  - fig4_trajectories.png   fp64 RK4 tracks of several momenta through the magnet
  - fig5_kick_integral.png  tx(z) and cumulative bending power for p=10 GeV
  - numbers.txt             key scalars cited in the Notion chapter

The trajectory integrator is a clean fp64 RK4 that mirrors Allen's ODE
(ExtrapolatorCommon.cuh:46-54) and trilinear field access
(MagneticField.cuh:38-78).  It is the recommended *truth* generator.

NOTE on units: the stored field is in Gaudi units (tesla = 1e-3).  The state
'qop' fed to the ODE is c*q/p = 0.299792458*q/p[1/GeV].  Their product gives
dtx/dz directly in 1/mm (the locked kappa = 1e-3*qop contract).
"""
import struct, array, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELD = "/data/bfys/gscriven/TE_stack/Allen/input/detector_configuration/magfield.bin"
C_LIGHT = 0.299792458  # State.qop = C_LIGHT * q/p[1/GeV]

# ---------------------------------------------------------------- load field
raw = open(FIELD, "rb").read()
invD = struct.unpack_from("<3f", raw, 0)
N = struct.unpack_from("<3i", raw, 16)
mn = struct.unpack_from("<3f", raw, 32)
Nx, Ny, Nz = N
minX, minY, minZ = mn
dx, dy, dz = 1/invD[0], 1/invD[1], 1/invD[2]
flat = array.array("f"); flat.frombytes(raw[48:48 + Nx*Ny*Nz*4*4])
B = np.array(flat, dtype=np.float64).reshape(Nz, Ny, Nx, 4)  # [iz,iy,ix,(Bx,By,Bz,pad)]
Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
print(f"grid {Nx}x{Ny}x{Nz}, spacing ({dx:.0f},{dy:.0f},{dz:.0f}) mm, "
      f"min ({minX:.0f},{minY:.0f},{minZ:.0f})")

def field(x, y, z):
    """Allen trilinear lookup; returns (Bx,By,Bz) in Gaudi units, 0 outside grid."""
    fx = (x - minX)*invD[0]; fy = (y - minY)*invD[1]; fz = (z - minZ)*invD[2]
    ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
    if ix < 0 or iy < 0 or iz < 0 or ix >= Nx-1 or iy >= Ny-1 or iz >= Nz-1:
        return 0.0, 0.0, 0.0
    hx, hy, hz = fx-ix, fy-iy, fz-iz
    out = []
    for comp in (Bx, By, Bz):
        c = comp
        v = ((1-hz)*((1-hx)*(1-hy)*c[iz,iy,ix]   + hx*(1-hy)*c[iz,iy,ix+1]
                    + (1-hx)*hy*c[iz,iy+1,ix]    + hx*hy*c[iz,iy+1,ix+1])
              + hz *((1-hx)*(1-hy)*c[iz+1,iy,ix] + hx*(1-hy)*c[iz+1,iy,ix+1]
                    + (1-hx)*hy*c[iz+1,iy+1,ix]  + hx*hy*c[iz+1,iy+1,ix+1]))
        out.append(v)
    return out[0], out[1], out[2]

def deriv(s, qop):
    """Allen ODE: s=(x,y,tx,ty); returns d/dz."""
    x, y, tx, ty = s
    bx, by, bz = field(x, y, s_z[0])
    norm = np.sqrt(1 + tx*tx + ty*ty)
    ax = norm*(ty*(tx*bx + bz) - (1+tx*tx)*by)
    ay = norm*(-tx*(ty*by + bz) + (1+ty*ty)*bx)
    return np.array([tx, ty, qop*ax, qop*ay])

s_z = [0.0]  # current z (closure) so deriv can sample the field at the stage z

def rk4_track(p_gev, q, z0, z1, h=5.0, x0=0.0, y0=0.0, tx0=0.0, ty0=0.0):
    """fp64 RK4 from z0 to z1; returns z[], state[:,4]."""
    qop = C_LIGHT * q / p_gev
    s = np.array([x0, y0, tx0, ty0], dtype=np.float64)
    zs = [z0]; ss = [s.copy()]
    z = z0; n = int(round((z1-z0)/h))
    for _ in range(n):
        s_z[0] = z;        k1 = deriv(s, qop)
        s_z[0] = z+h/2;    k2 = deriv(s + h/2*k1, qop)
        s_z[0] = z+h/2;    k3 = deriv(s + h/2*k2, qop)
        s_z[0] = z+h;      k4 = deriv(s + h*k3, qop)
        s = s + h/6*(k1 + 2*k2 + 2*k3 + k4); z += h
        zs.append(z); ss.append(s.copy())
    return np.array(zs), np.array(ss)

# ----------------------------------------------------- key scalars / numbers
zc = np.arange(minZ, minZ+Nz*dz, dz)
ix0 = int(round((0-minX)*invD[0])); iy0 = int(round((0-minY)*invD[1]))
By_axis = By[:, iy0, ix0]
ipk = int(np.argmin(By_axis))              # most negative (MagDown)
z_peak = minZ + ipk*dz
# Bending power int By dz over UT->T (z=2665..7826), trapezoid on the 100mm grid:
m = (zc >= 2665) & (zc <= 7826)
_trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
intBydl_Tm = _trap(By_axis[m], zc[m])   # stored*mm == T*m  (tesla=1e-3, mm/1000)
pT_kick = C_LIGHT * abs(intBydl_Tm)        # GeV
numbers = {
    "grid": [Nx, Ny, Nz], "spacing_mm": [dx, dy, dz],
    "By_axis_at_z5000_gauss_units": float(By[int(round((5000-minZ)*invD[2])), iy0, ix0]),
    "By_peak_value_gaudi": float(By_axis[ipk]),
    "By_peak_value_tesla": float(By_axis[ipk]*1000),
    "By_peak_z_mm": float(z_peak),
    "intBydl_UTtoT_Tm": float(intBydl_Tm),
    "pT_kick_GeV": float(pT_kick),
}
for p in (3, 5, 10, 20, 50):
    z, s = rk4_track(p, +1, 2665.0, 9400.0, h=5.0)
    numbers[f"deflection_dx_mm_p{p}"] = float(s[-1,0])
    numbers[f"final_tx_p{p}"] = float(s[-1,2])
open("numbers.txt", "w").write(json.dumps(numbers, indent=2))
print(json.dumps(numbers, indent=2))

# ------------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
regions = [("VELO", -300, 800, "#cfe8ff"), ("UT", 2327, 2642, "#d8f5d0"),
           ("magnet", 2900, 7800, "#ffe3c2"), ("SciFi (T)", 7826, 9403, "#f3d0f0")]

# Fig 1: By along axis
fig, ax = plt.subplots(figsize=(8, 4))
for name, za, zb, col in regions:
    ax.axvspan(za, zb, color=col, alpha=0.6, label=name)
ax.plot(zc, By_axis*1000, "k-", lw=1.6)
ax.axvline(z_peak, color="r", ls="--", lw=0.9)
ax.set_xlabel("z [mm]"); ax.set_ylabel(r"$B_y(0,0,z)$ [T]")
ax.set_title(f"LHCb dipole on axis (v8r1 down):  peak $B_y$ = {By_axis[ipk]*1000:.3f} T @ z = {z_peak:.0f} mm")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("fig1_By_axis.png"); plt.close(fig)

# Fig 2: By(x,0,z) slice
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = minX + np.arange(Nx)*dx
im = ax.pcolormesh(zc, xs, (By[:, iy0, :].T)*1000, cmap="RdBu", shading="auto",
                   vmin=-1.2, vmax=1.2)
ax.set_xlabel("z [mm]"); ax.set_ylabel("x [mm]")
ax.set_title(r"$B_y(x,\,y{=}0,\,z)$ [T] — the bending component")
fig.colorbar(im, ax=ax, label="T"); fig.tight_layout()
fig.savefig("fig2_By_xz_slice.png"); plt.close(fig)

# Fig 3: B components along axis
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(zc, Bx[:, iy0, ix0]*1000, label=r"$B_x$")
ax.plot(zc, By_axis*1000, label=r"$B_y$")
ax.plot(zc, Bz[:, iy0, ix0]*1000, label=r"$B_z$")
ax.set_xlabel("z [mm]"); ax.set_ylabel("B [T]")
ax.set_title("Field components on axis (Bx,Bz ~ 0; By dominates)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("fig3_B_components.png"); plt.close(fig)

# Fig 4: trajectories
fig, ax = plt.subplots(figsize=(8, 4.4))
for name, za, zb, col in regions:
    ax.axvspan(za, zb, color=col, alpha=0.5)
for p in (3, 5, 10, 20, 50):
    z, s = rk4_track(p, +1, 2665.0, 9400.0, h=5.0)
    ax.plot(z, s[:, 0], label=f"p = {p} GeV")
ax.plot([2665, 9400], [0, 0], "k:", lw=0.8, label="straight line")
ax.set_xlabel("z [mm]"); ax.set_ylabel("x [mm]")
ax.set_title("fp64 RK4 tracks (q=+1) from UT exit through the magnet into SciFi")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("fig4_trajectories.png"); plt.close(fig)

# Fig 5: tx(z) + cumulative bending power for p=10 GeV
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
z, s = rk4_track(10, +1, 2665.0, 9400.0, h=5.0)
a1.plot(z, s[:, 2], "b-"); a1.set_ylabel(r"$t_x = dx/dz$")
a1.set_title("p = 10 GeV: slope kick accumulates where By is strong"); a1.grid(alpha=0.3)
cum = np.concatenate([[0], np.cumsum(0.5*(By_axis[:-1]+By_axis[1:])*dz)])  # int By dz (Gaudi*mm = T*m)
a2.plot(zc, -cum*C_LIGHT, "g-"); a2.set_ylabel(r"$0.3\!\int_{}^{z}\!(-B_y)\,dl$  [GeV]")
a2.set_xlabel("z [mm]"); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig5_kick_integral.png"); plt.close(fig)

# Fig 6: detector schematic + Allen stage map -----------------------------
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(12.5, 6.4))
ax.set_xlim(-750, 10350); ax.set_ylim(-6.5, 4.2); ax.axis("off")
ax.annotate("", xy=(10300, 0), xytext=(-750, 0), arrowprops=dict(arrowstyle="->", color="0.55"))
ax.text(10300, 0.22, "z [mm]", fontsize=9, color="0.4", ha="right")

def det(z0, z1, h, name, color, fs=9, alpha=0.85):
    ax.add_patch(mpatches.Rectangle((z0, -h), z1-z0, 2*h, fc=color, ec="k", lw=0.7, alpha=alpha))
    ax.text((z0+z1)/2, h+0.08, name, ha="center", va="bottom", fontsize=fs)

det(-300, 800, 1.6, "VELO\n(21 layers)", "#bcd6f7")
det(1000, 2150, 1.0, "RICH1", "#ededed", fs=7, alpha=0.6)
for z in (2327.5, 2372.5, 2597.5, 2642.5):
    ax.add_patch(mpatches.Rectangle((z-7, -1.5), 14, 3.0, fc="#9ad48f", ec="k", lw=0.5))
ax.text(2485, 1.65, "UT (4 layers)", ha="center", va="bottom", fontsize=9)
ax.add_patch(mpatches.Rectangle((2900, -2.7), 4900, 5.4, fc="#ffe3c2", ec="k", lw=0.7, alpha=0.5))
ax.text(5350, 2.85, "DIPOLE MAGNET", ha="center", va="bottom", fontsize=10, weight="bold")
mmag = (zc >= 2300) & (zc <= 8000)
ax.plot(zc[mmag], By_axis[mmag]*2.0, color="#a01010", lw=1.8)
ax.text(4700, By_axis[ipk]*2.0-0.28, "By(z), peak -1.05 T @ 4700", color="#a01010", ha="center", fontsize=8)
for lab, (za, zb) in [("T1", (7826, 8036)), ("T2", (8508, 8718)), ("T3", (9193, 9403))]:
    for z in np.linspace(za, zb, 4):
        ax.add_patch(mpatches.Rectangle((z-7, -1.7), 14, 3.4, fc="#e7b3e0", ec="k", lw=0.4))
    ax.text((za+zb)/2, 1.85, lab, ha="center", va="bottom", fontsize=8)
ax.text(8614, 2.55, "SciFi (T1,T2,T3 = 12 layers)", ha="center", va="bottom", fontsize=9)

# vertical guide lines linking detector boundaries to the stage lanes
for z in (800, 2327, 2642, 7826, 9403):
    ax.plot([z, z], [0, -6.0], color="0.85", lw=0.7, ls=":", zorder=0)

# one stage per row (no label collisions)
def stage(z0, z1, y, label, color):
    ax.plot([z0, z1], [y, y], color=color, lw=3.0, solid_capstyle="butt")
    for z in (z0, z1):
        ax.plot([z, z], [y-0.13, y+0.13], color=color, lw=3.0)
    ax.text((z0+z1)/2, y+0.20, label, ha="center", va="bottom", fontsize=7.7, color=color)

ax.text(-750, -2.55, "Allen stages (each row = one extrapolation step):", fontsize=9, weight="bold", color="0.2")
stage(-300, 800, -3.05, "VELO Kalman fwd+bwd:  CreateVeloSeedState + ExtrapolateInV", "#1f5fbf")
stage(800, 2327, -3.60, "ExtrapolateVUT  (VELO -> first UT layer)", "#159a4e")
stage(2327, 2642, -4.15, "ExtrapolateInUT  x3", "#159a4e")
stage(2642, 7826, -4.70, "PredictStateUTT (UT -> T, crosses magnet):  InUT->2665 | extrapUTT 2665->7826 | TFT", "#d07000")
stage(7826, 9403, -5.25, "ExtrapolateInT  x11", "#9b2fae")
stage(2665, 9400, -5.85, "extrapolate_states : Cash-Karp RK -- samples the field map, ~600 lookups/state (SV reco)", "#a01010")
ax.set_title("LHCb tracking detector along z, and where each Allen extrapolation stage lives",
             fontsize=11.5)
fig.tight_layout(); fig.savefig("fig6_detector_schematic.png", dpi=130); plt.close(fig)

print("\nFigures written to", __file__.rsplit("/",1)[0])
