# TX_project

Scoping of **track extrapolation in LHCb Allen** (GPU/HLT1) — figures and a
reference integrator that back the Notion write-up
*Allen Track Extrapolation — Reference Chapter*.

See [`Allen_scoping/`](Allen_scoping/):

- `field_and_trajectories.py` — reads the Allen field map and is a clean fp64 RK4
  reference integrator (the recommended truth generator for surrogate data-gen).
- `fig1..fig6` — dipole profile, field slice/components, RK4 trajectories, the
  cumulative bending power, and a **detector-along-z schematic with the Allen
  extrapolation-stage map**.
- `numbers.txt` — key scalars (peak field, ∫B·dl, pT kick, deflections).
- `README.md` — figure index, Mermaid diagram sources, locked conventions.

All figures derive from public LHCb field-map data and toy trajectories.
