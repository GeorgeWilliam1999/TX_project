# 02 — Timeline

All dates 2026. The project ran in two eras separated by the κ discovery on 06-11.

## Era 1 — the weak-field era (looked like success)
- **gen-1, gen-2** — early NN architectures (MLP, then physics-informed variants). Established the
  pipeline: RK4 ground-truth corpus, training, evaluation. Resized for the 64 kB Allen constant-memory
  budget. Reported micron-scale accuracy.
- **gen-3** — the **PINN_v2** architecture: a ζ-gated correction envelope, IC exact by construction,
  PDE residual in the training loss (forward-mode JVP). Headline: ~12 µm median, ~293 µm on UT→T;
  locked candidate `pinn_v2_ALLEN_v1` (10,372 params, 40 kB). Exported to a V3 blob + generated CUDA
  header (`PINN_V2_UTT.cuh`) and wired into Allen's UT→T Kalman step (behind `m_use_nn_utt`).
- Metric/methodology work: log-cosh + median selection (Fix L1), 7-D input with `log10|dz|`/`sign(dz)`
  (Fix M1), the fp64 A4 Jacobian gate.
- **The kick-scaled head** (06-09/10): a correction scaled by κ·Δz so the q/p magnitude is exact by
  construction. Halved the median but worsened the low-p tail; the 10M run showed the tail was
  *structural*, motivating a λ=0 ablation.

## The pivot — 06-11
- **Restructure & repos:** split into the lab (`TrackExtrapolation`) and the deliverable
  (`track-extrapolation-pinn`, GitHub); Allen kept pristine; a Fable subagent did the
  `core/charts/gates` repo layout.
- **The κ / field discovery (P0.1 bake-off):** the first comparison against the production extrapUTT
  polynomial revealed the corpus magnetic coupling was **κ=1e-6 where physics needs 1e-3 — ×1000 too
  weak**, plus a sign-flipped polarity and the wrong field map. All gen-1→3 accuracy was a weak-field
  artifact. (See `03_kappa_and_field_discovery.md`.)
- **The fix & calibration:** κ→1e-3; canonical field = LHCb **FieldMap v8r1 down** (the CVMFS file
  Allen consumes); m_polarity=−1. The corrected stack reproduces extrapUTT to **15 µm** — external
  validation.

## Era 2 — the physical era (honest results)
- **gen-4 corpus** (06-11): regenerated at physical κ + v8r1 + 70/30 pointing population; 9.19M tracks;
  passed integrity/physics/population gates.
- **Three-arm eval** (06-14): on the real field, the gen-4 NNs are ~175 mm on UT→T — barely beating the
  225 mm straight line, ~16,000× worse than extrapUTT's 11–15 µm.
- **Parallel close-out (06-14/15), three workstreams:**
  - **A4 reference** rebuilt at physical κ (the old one was weak-field, invalid).
  - **Speed benchmark:** the NN is **slower than RK (1.2×) and the polynomial (3×)**; the deployed
    hybrid is strictly slower. (See `05_speed_benchmark.md`.)
  - **Wave-2 retraining:** data restratified (UT→T 0.145%→23%), residual head, capacity sweep
    h32→h384, proper training. UT→T error 175 mm → ~3 mm (a 55× data-fix gain) but **plateaus at
    ~3 mm regardless of size** — a real floor, ~285× worse than extrapUTT. **NN accuracy route closed.**
  - **Chart at physical κ** (06-14, separate effort): rebuilt on v8r1; **also loses to extrapUTT,
    ~3.9–4.7 mm vs 15 µm (~260×)** — the global multipole can't fit the real field's localised
    structure. (See `06_chart_programme.md`.)

## End of v1 — 06-16
Both surrogate routes shown non-competitive on the real field. Project archived here and re-scoped
end-to-end from Allen. Literature retained; chart code carried forward in `../Chart/`.
