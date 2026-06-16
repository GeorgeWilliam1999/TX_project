# F4 — Allen extrapUTT bake-off + ADR 0011 (status: started 2026-06-11)

## ADR 0011 — drafted (Proposed)
`docs/decisions/0011-analytic-chart-admissibility.md`. Argues the chart is admissible
under ADR 0009 by **precedent**: the production incumbent `extrapUTT`
(`Allen/device/kalman/ParKalman/include/ParKalmanMethods.cuh:287`) is itself a
frozen-coefficient analytic kick map — order-2 even-multipole bend
(`bendx = BENDX + BENDX_X2·(x/z)² + BENDX_Y2·(y/z)²`, `bendy = BENDY_XY·(x/z)(y/z)`),
qop kick `fq = qop·PMIN`, binned-polynomial coefficient tables (`kalman_params->x00…`).
The chart is the same class at higher fidelity (O12 vs O2). ADR 0009's ban targets the
3-D field map + adaptive RK loop, which the chart does not use. Ruling left to G. Scriven;
conditions: A4 gate, 63.5 kB counts against the weight budget, single fixed-cost pass.

## Bake-off — BLOCKED on a broken incumbent harness
The existing P0.1 scoring (`gen_3/paper_p0/P0p1_baseline_verdict.json`) is **nonphysical**:

| candidate | median dx | p95 | p99 |
|---|---|---|---|
| extrapUTT polm1 | **369,801 µm (369 mm)** | 401 m | 10 km |
| extrapUTT polp1 | 362,923 µm | 501 m | 12 km |
| straight_line | 113.7 µm | 2325 | 4739 |
| pinn_v2_small_v1 | 251.9 µm | 2232 | 4536 |

extrapUTT producing a 369 mm median (with km-scale tails) on a pool where a straight line
gives 114 µm means the standalone polynomial harness is mis-wired or evaluated outside its
valid (p, angle) domain — NOT a usable incumbent baseline. (Note this pool/metric also
differs from the flattening UT→T pool, where the locked NN is 293 µm and straight-line is
mm-scale — the P0.1 "plane" reference uses extrapUTT's native z_i→z_f planes, not the corpus
mask.) A clean bake-off needs the harness fixed first.

## F4a root-cause diagnosis (2026-06-11)
The bug is in `poly_pred_pol{m1,p1}.csv` (the pre-computed extrapUTT outputs that
`compare_baseline.py` just diffs against `Y_true` on the z 2665→7826 plane). The poly
predictions are **decorrelated** from truth, not merely scaled:
- On the 2408 tracks with a real bend (|true bend_x|>0.5mm), the ratio poly_bend/true_bend
  has median −767 (x) / −965 (tx) but IQR [−2178, +370] — i.e. broad and **sign-indefinite**.
- A pure corpus-qop units error (299.792458× factor) would give a tight ratio of 1.0; it gives 6.2.
So it is **not a one-line scalar fix** — it is a structural wiring error in the extrapUTT
evaluation that produced the CSVs (input-grid normalisation of ux,uy,fq via Txmax/Tymax/PMIN,
the binned-polynomial cell indexing ix,iy, or the loaded `kalman_params` coefficients).
Fixing it = reproduce extrapUTT (`ParKalmanMethods.cuh:287`) correctly in the harness, which
needs the actual `KalmanParametrizations` coefficient files (x00/x10/x01/tx00…) + dev_UTT_META.

**Feasibility (06-11): the coefficient files EXIST locally** — `read_params_UTT()` in
`KalmanParametrizations.cuh` loads `ParametrizedKalmanFit/25v0/params_UTT_v0.tab`, present at:
- `TE_stack/PARAM/ParamFiles/data/ParametrizedKalmanFit/25v0/params_UTT_v0.tab` (the 25v0 set
  `SetParameters` points to), and a `24v0/MagDown/params_UTT_v0.tab` matching our polarity.
So F4a is **feasible locally** — the only blocker is the reimplementation effort (parse the
`.tab`, replicate `compute_state<DEGx2,DEGx1>` binned-polynomial eval + the ux,uy,fq
normalisation), not missing data. The `.tab` format/parser is `read_params_UTT` (declared
`KalmanParametrizations.cuh:63`); dev_UTT_META is a `__constant__` (`ParKalmanSharedConstants.cuh:26`).

## Next (F4 todos)
- F4a: reimplement/validate extrapUTT in the harness against the loaded coefficient tables;
  reproduce a SANE incumbent number on the plane reference (structural fix, not a scale tweak).
- F4b: once F4a is sane, publish the comparison table (straight / chart / extrapUTT / NN)
  on the common UT→T pool, and run the chart through the A4 Jacobian gate.
- F4c: ADR 0011 ruling (draft ready).
