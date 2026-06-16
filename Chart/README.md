# Analytic Chart — Track Extrapolation by Phase-Space Flattening

**Self-contained snapshot, 2026-06-16.** This is the one v1 direction worth carrying into the rescope.
A full Notion write-up is coming soon; this README is the working anchor until then.

> **Honest status up front:** on the *real* LHCb field the current chart **loses to the production
> extrapUTT polynomial ~260×** (chart ~3.9–4.7 mm vs extrapUTT 15 µm). Its celebrated 12.1 µm result
> was an artifact of the smooth *toy* field. The flattening *principle* is sound; the *global-multipole
> field representation* is the weak link. A **localised** representation is the only path that might
> make it competitive — see "Where it stands" below. Do not deploy the chart as built.

## The idea (one paragraph)
RK extrapolation is expensive because the dipole field makes phase-space trajectories curved. Instead
of learning the curved endpoint map (the neural-net route, now closed), we **analytically flatten** the
curvature: for an on-axis field `B=By(z)ŷ` the canonical momentum `Px = px + q·F(z)` (with `F=∫By dz`)
is conserved and the field-weighted measure τ(z)∝F(z) linearises the leading dynamics. Inference uses
small frozen tables `c_ab(z)` + a fixed chord quadrature — **no field map, no RK loop at inference.**
The real field's transverse structure is the only hard part.

## Where it stands (results)
| candidate (real v8r1 field, PV-pointing, UT→T) | median dx | p95 |
|---|---|---|
| straight line | 237 mm | 1.37 m |
| chart O12 rebuilt on v8r1 | 4,725 µm | 137 mm |
| chart best-tuned (O20) | 3,868 µm | — |
| **extrapUTT (incumbent)** | **14.9 µm** | 2.2 mm |
| chart on the *toy* field (historical) | 12.1 µm | (not deployment-relevant) |

**Root cause:** a global degree-≤20 even-multipole basis fits the toy By to ~2 % but real v8r1 to only
~8 % (Runge ringing at the window edges; the real field has localised fringe structure, raw peak
4.94 T). extrapUTT wins because it fits a degree-9 polynomial **per 60×50 (x,y) bin** — locally.

**The one open path:** a localised field representation. A coarse 3-D grid + trilinear cut the median
field-reconstruction error ~10× (0.37 % at 52 kB, ~18× smaller than extrapUTT's 960 kB tables), which
would likely move the chart median toward ~0.1–1 mm. **Parity with 15 µm is NOT demonstrated**, and a
localised chart converges toward extrapUTT's own design. Bar for the rescope: **match ~15 µm at
≪960 kB, or don't deploy.**

## Layout
```
PLAN.md                     full programme plan (F0–F6) + TL;DR
charts/                     the chart itself
  build_field_integrals.py    F0: F(z),G(z) integral tables + κ0 calibration
  chart.py / chart_multipole.py  the multipole chart (toy-field build)
  build_chart_v8r1.py         the physical-field rebuild  ->  chart_tables_v8r1.npz
  *.npz                       small frozen tables (chart_tables, *_v8r1, field_integrals, multipole)
  make_residual_corpus.py     (residual-NN sub-line, CLOSED)
benchmarks/                 ladder + diagnostic scripts (ladder_utt, rung2, sweep_weight, ...)
results/                    the F-phase notes & json — READ F4b_bakeoff_v8r1 + F4a first
core/                       extracted dependencies (field_v8r1, magnetic_field, rk4_propagator, architectures)
paper_p0/                   the extrapUTT yardstick + truth set for the bake-off
  extraputt_py.py             faithful Python port of Allen's extrapUTT (validated bit-faithful)
  v8r1_plane_truth.npz        PV-pointing RK truth @ z 2665->7826 (v8r1, κ=1e-3)
  plane_poly_v8r1_polm1.csv   extrapUTT predictions on that set
docs/decisions/0011-...     ADR: are frozen tables admissible under the replacement criterion?
trained_models/             the (closed) residual-NN checkpoints, tiny
```

## How to run
- **Python:** `/data/bfys/gscriven/conda/envs/TE/bin/python` (numpy; torch only for the residual-NN
  benchmarks).
- **Dependencies note:** the chart scripts import `field_v8r1`, `magnetic_field`, `rk4_propagator`
  (and `architectures` for the residual bench), which are bundled in `core/`. Put `core/` on the path,
  e.g. `PYTHONPATH=core python charts/build_chart_v8r1.py` — paths may need minor adjustment since this
  is an extracted snapshot.
- **The key reproduction:** `results/bake_v8r1.py` → the chart-vs-extrapUTT-vs-straight table above,
  using `paper_p0/v8r1_plane_truth.npz` (truth) and `charts/chart_tables_v8r1.npz` (the chart).
- **Field map (read-only, external):** `/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.down.bin`.

## Conventions (locked)
κ = 1e-3·qop, qop = 0.299792458·q/p[1/GeV]; field = v8r1 down (raw sign, MagDown By<0); extrapUTT pairs
with m_polarity = −1. `core/field_v8r1.py` returns −By (raw); the legacy `get_field_numpy` returns +By
(toy, ×1000 scaling) — the v8r1 chart's κ0 and tables are built consistently from `FieldV8R1`.

## Not included (regenerable, too big)
The residual training corpora (`residual_2M.npz` 146 MB, `residual_fwd.npz` 12 MB) and the gen-4
corpus. Regenerate via `charts/make_residual_corpus.py` if the residual sub-line is ever revisited.

## What a Notion write-up should cover
The flattening theory (canonical momentum + τ-measure), the toy→real collapse and its root cause, the
localised-representation feasibility probe, the ADR-0011 admissibility argument, and the explicit
go/no-go bar (match extrapUTT 15 µm at ≪960 kB). See `../archive/06_chart_programme.md` for the v1
record this builds on.
