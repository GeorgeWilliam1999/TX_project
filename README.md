# TX_project — LHCb Track Extrapolation (consolidated)

Working repository for the LHCb **track-extrapolation** effort (GPU/HLT1): replacing /
augmenting the Runge–Kutta extrapolator with compact surrogates, and the supporting
physics, data and analysis. Backs the corresponding Notion write-ups.

## Layout

- **[`Allen_scoping/`](Allen_scoping/)** — how track extrapolation works in Allen:
  `field_and_trajectories.py` (reads the Allen field map; clean fp64 RK4 reference
  integrator — the recommended truth generator for surrogate data-gen), figures
  (dipole profile, field slices/components, RK4 trajectories, cumulative bending power,
  detector-along-z schematic), `numbers.txt`, and conventions. Backs the Notion
  *Allen Track Extrapolation — Reference Chapter*.
- **[`field_map/`](field_map/)** — end-to-end characterisation of the **verified** field
  `field.v8r1.down.bin` (= `v5r11`, the map Allen loads and extrapUTT was fit to):
  loader `field_v8r1.py`, two executed notebooks (identity/geometry/on-axis/polarity;
  transverse structure/multipole ceiling/Maxwell checks), `figures/`, and `PLAN.md`.
  Backs the Notion *LHCb Magnetic Field Map (v8r1) — End-to-End Characterisation*.
- **[`Chart/`](Chart/)** — the **analytic-flattening** research line (phase-space charts):
  chart builders + small frozen tables, benchmarks, the F-phase result notes
  (`results/F4b_…` is the verdict), and the extrapUTT yardstick. Honest status: the
  global-multipole chart loses to extrapUTT on the real field; a localised redesign is
  the open path (`README.md` has the go/no-go bar).
- **[`archive/`](archive/)** — the durable record of the prior generation (v1): timeline,
  the κ/field discovery, the neural-net results, the speed benchmark, the chart programme,
  data schemas, and the verdict & lessons. Read `archive/README.md` first.

## Locked conventions (used throughout)

`κ = 1e-3·qop`, `qop = 0.299792458·q/p[1/GeV]` (= Allen c·q/p); field = **v8r1 down**
(raw MagDown By<0; the CVMFS `field.v8r1.down.bin` Allen consumes); production `extrapUTT`
pairs with `m_polarity = −1`; RK4 truth at 5 mm step (validated vs extrapUTT to 15 µm).

All figures derive from public LHCb field-map data and toy/simulated trajectories.
