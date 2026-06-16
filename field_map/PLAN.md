# Field-Map Characterisation — Plan

**Dir:** `/data/bfys/gscriven/Ex_rep/field_map/` · **Created:** 2026-06-16
**Purpose:** lock down *which* magnetic field our data generation and training must use, prove it is
the right one, then characterise it end-to-end (notebooks → Notion write-up) so every downstream
choice (corpus, chart target, RK truth) rests on a verified foundation.

## 0. The verified answer (which field)
**Use `field.v8r1.down.bin` (= `v5r11/cdf/field.v8r1.down.bin`).** Evidence (checked 2026-06-16):
- `v8r1` is a **symlink to `v5r11`** — the latest production field map; nothing newer exists on CVMFS.
- Allen's `detector_configuration` and **every 2025 production geometry** load this exact file as
  `magfield.bin`. It is the map `extrapUTT` was fit to (validated to 15 µm / 0.17% in F4a/F4b).
- A sign-flipped twin **`field.v8r1.up.bin`** exists (one MagUp geometry uses it). Production runs
  **both polarities**; down is the default. `By(0,0,5000)`: down −1.034 T, up +1.034 T (exact flip).

**Consequence for data:** generate against **v8r1 down** with κ = 1e-3·qop, qop = 0.299792458·q/p[1/GeV],
raw sign (MagDown By<0). The corpus/surrogate must eventually handle **both polarities** (the up map
is the down map negated) — record this as a first-class requirement, not an afterthought.

## 1. Deliverables
- `field_v8r1.py` — the validated loader (copied; decodes the Allen binary, auto-scales Gaudi→Tesla).
- **Notebooks** (`*.ipynb`, executed, figures in `figures/`):
  - `01_identity_geometry_onaxis.ipynb` — provenance, binary format, grid geometry, units & sign,
    on-axis B_y(z) profile, field integral ∫B_y dz + effective bending, down-vs-up polarity check,
    coverage vs the LHCb track region.
  - `02_transverse_multipole_maxwell.ipynb` — 2-D B_y slices, off-axis (transverse) structure,
    fringe B_x/B_z, **multipole decomposition vs order (the global-fit accuracy ceiling / Runge
    phenomenon — why the global-multipole chart failed)**, numerical Maxwell checks (∇·B, ∇×B),
    the field actually traversed by PV-pointing tracks (incl. the raw peak |B|), and the contrast
    with the (wrong) toy `twodip` field.
- **Notion write-up** built from the notebook outputs — a full, end-to-end exploration of the field
  map (identity → geometry → on-axis → transverse → multipole → Maxwell → implications for data).

## 2. Characteristics to quantify (the checklist)
1. Identity & provenance: file, version (v8r1=v5r11), polarity, that it is Allen's `magfield.bin`.
2. Binary format: header (invDxyz | Nxyz | minXYZ | N×4 floats), units (Gaudi → ×1000 → Tesla).
3. Grid geometry: N=(81,81,146), 100 mm voxels, extent x,y∈[−4000,4000], z∈[−500,14000] mm.
4. Sign & polarity: MagDown B_y<0; down = −up (verify max|B_down+B_up|≈0).
5. On-axis B_y(0,0,z): the dipole bump, peak value & z, the field integral, effective bend per GeV.
6. Transverse structure: B_y(x,y) growth off-axis; the localised structure (non-monotonic, peak ~5 T).
7. Fringe components B_x, B_z off-axis (Maxwell-forced).
8. Multipole content: even-only basis, fit RMS vs order over the chord window; the ~8% ceiling.
9. Maxwell consistency: numerical ∇·B≈0, ∇×B≈0 in the tracking (current-free) volume.
10. Track-traversed region: PV-pointing chord coverage, max |B| seen, the worst region for surrogates.
11. Comparison to the toy `twodip` (the field v1 wrongly used): on-axis 1.4% weaker + sign flip →
    473 mm UT→T endpoint divergence (from archive).
12. Implications: data-generation field/convention, RK truth step size, polarity handling.

## 3. Execution
1. Build + run both notebooks (`jupyter nbconvert --execute`), figures → `figures/`. ✅ this pass.
2. Mirror figures to the deliverable repo `docs/figures/` (so Notion can render them after push).
3. Create the Notion write-up (Write-Up DB, tagged to the project) with the quantitative findings,
   tables, math, and the figures.

## 4. Open questions for the rescope (recorded, not answered here)
- Train down-only and negate for up, or train polarity-agnostic? (The EOM is linear in B, so a
  down-trained map negated should serve up — verify.)
- Is the 100 mm grid + trilinear (Allen's own access) the truth we should match, or a finer RK on
  the underlying `.cdf` parametrisation? (Allen interpolates the 100 mm grid; matching that is the
  honest target.)
- Which sub-region matters most for the chosen extrapolation target (UT→T vs general steps)?
