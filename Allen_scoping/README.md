# Allen Track Extrapolation — scoping working folder

Companion artifacts for the Notion reference chapter
**Allen Track Extrapolation — Reference Chapter (2026-06, illustrated)**
<https://app.notion.com/p/3815d544b9d981188488dda13d9b9d4b>

Everything here was produced READ-ONLY from `/data/bfys/gscriven/Allen` and
`/data/bfys/gscriven/TE_stack` (nothing in those trees was modified).

## Files

| file | what it is |
|---|---|
| `field_and_trajectories.py` | reads `magfield.bin`, makes the figures, and is a clean **fp64 RK4 reference integrator** (the recommended truth generator) |
| `numbers.txt` | key scalars (peak field, ∫B·dl, pT kick, deflections) as JSON |
| `fig6_detector_schematic.png` | **detector along z + which Allen stage lives where** (VELO/UT/magnet/SciFi boxes, By profile, one labelled bracket per stage) |
| `fig1_By_axis.png` | By(0,0,z) dipole profile over the full grid, detector regions shaded |
| `fig2_By_xz_slice.png` | By(x, y=0, z) heat map — the bending component in the x–z plane |
| `fig3_B_components.png` | Bx, By, Bz on axis (Bx,Bz≈0; By dominates) |
| `fig4_trajectories.png` | fp64 RK4 tracks for p = 3,5,10,20,50 GeV from UT exit through the magnet |
| `fig5_kick_integral.png` | tx(z) and cumulative bending power for p = 10 GeV |

Regenerate everything: `python3 field_and_trajectories.py`

## Key numbers (numbers.txt)

- field grid `81 × 81 × 146`, isotropic `100 mm`, min `(-4000,-4000,-500)` mm
- peak `By = -1.048 T @ z = 4700 mm` (MagDown, By<0); units are Gaudi (`tesla = 1e-3`)
- `∫By·dl` over UT→T (z 2665→7826) `= -3.733 T·m`  ⇒  `pT kick = 0.299792458·|∫B·dl| = 1.12 GeV`
- deflection scales as `Δx ∝ tx_out ≈ pT_kick / p` (e.g. 10 GeV → Δx≈484 mm, tx≈0.121)

## Conventions (locked)

- `kappa = 1e-3 · qop`, `qop = 0.299792458 · q/p[1/GeV]` (= Allen `c·q/p`)
- field = LHCb FieldMap **v8r1 down**, raw MagDown `By < 0`, no sign flips
- `extrapUTT` pairs with `m_polarity = -1`

## Mermaid diagram sources (also embedded in the Notion page)

### 1 — Where extrapolation sits in HLT1
```mermaid
flowchart TD
  raw["Raw banks"] --> velo["VELO tracks"] --> ut["Velo+UT tracks"] --> long["SciFi -> Long tracks (Velo+UT+T)"]
  long --> kf["kalman_filter_t : forward+backward Kalman fit<br>parametrised, FIELD-FREE, ~60 steps/track"]
  kf --> states["FittedTrack + KalmanStates"]
  states --> sv["Secondary-vertex reconstruction"]
  sv --> rk["extrapolate_states_t : Cash-Karp RK<br>SAMPLES THE FIELD MAP, ~600 lookups/state"]
  kf -. downstream .-> dkf["DownstreamKalmanFilter (UT+T)"]
  classDef hot fill:#ffe3c2,stroke:#c47;
  class rk hot;
```

### 2 — Per-track Kalman fit sequence
```mermaid
flowchart TD
  seed["CreateVeloSeedState @ last VELO hit"] --> vf["VELO forward: Predict/Update x (nV-1)"]
  vf --> vut["ExtrapolateVUT -> first UT layer"]
  vut --> utl["ExtrapolateInUT x3 (UT layers 1-3)"]
  utl --> UTT
  UTT --> tl["ExtrapolateInT x11 (SciFi layers 1-11)"]
  tl --> back["Backward VELO pass: C <- inv(F_total) C"]
  back --> out["MakeTrack + propagate_to_beamline"]
  subgraph UTT["PredictStateUTT : composite, z 2642.5 -> 7826"]
    direction LR
    a["ExtrapolateInUT -> 2665"] --> b["ExtrapolateUTT : extrapUTT poly -> 7826"] --> c["ExtrapolateTFT -> first T plane"]
  end
```

### 3 — Cash–Karp RK step
```mermaid
flowchart LR
  s["state s, qop"] --> k0["B@s -> k0"] --> k1["B@(s+a*k) -> k1"] --> k2["-> k2"] --> k3["-> k3"] --> k4["-> k4"] --> k5["-> k5"]
  k5 --> comb["state += sum b_i k_i<br>err += sum (b_i - b*_i) k_i (DISCARDED)"]
```

### 4 — extrapUTT dataflow
```mermaid
flowchart TD
  in["x,y,tx,ty,qop @ z=ZINI=2665"] --> nrm["normalise: xx=x/(zi*Txmax), yy=y/(zi*Tymax)"]
  nrm --> bin["bin (ix,iy) + local gx,gy in [-0.5,0.5]"]
  in --> res["pointing residuals<br>ux=(tx - x/zi - bx*qop)/Dtxy<br>uy=(ty - y/zi - by*qop)/Dtxy"]
  in --> fq["fq = qop * PMIN"]
  bin --> itp["6-neighbour quadratic interp -> coeffs c00,c10,c01"]
  itp --> pol["state += sum_d (c00 + c10*ux + c01*uy) * fq^(d+1)"]
  res --> pol
  fq --> pol
  pol --> out["x,y,tx,ty @ z=ZFIN=7826  +  Jacobian der_*"]
```

### 5 — Field map load + lookup
```mermaid
flowchart TD
  binf["magfield.bin (15.3 MB on disk)"] --> hdr["parse 48-byte header: invD, N(int), min"]
  hdr --> de["de-interleave stream (Bx,By,Bz,pad) stride 4"]
  de --> dev{"MAGFIELD_USE_TEXTURE ?"}
  dev -->|no| arr["3 x float* device arrays (11 MB)<br>manual 8-point trilinear"]
  dev -->|yes| tex["3 x cudaTextureObject<br>hardware linear filter, tex3D"]
  arr --> look["fieldVectorLinearInterpolation(x,y,z)"]
  tex --> look
  look --> use["used ONLY by RK propagate (6x per step)"]
```
