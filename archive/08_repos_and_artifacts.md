# 08 — Repos & Artifacts (where everything lives)

## The four local trees
| Path | Role |
|---|---|
| `/data/bfys/gscriven/TrackExtrapolation` | **The lab** — all live work, data, checkpoints, mlruns. `experiments/gen_3` (NN line) and `experiments/flattening` (chart line). Big data (`*.npz`, `trained_models/`) is local-only. |
| `/data/bfys/gscriven/track-extrapolation-pinn` | **The deliverable** — curated GitHub repo `GeorgeWilliam1999/TrackExtrapolation_RKPINN`. Layout: `core/ models/ charts/ gates/ datagen/ allen_bridge/ For_Allen/ candidate/ docs/`. Scripts resolve lab data via `TE_LAB`. |
| `/data/bfys/gscriven/Allen` | **The Allen MR** — kept pristine (branch `gscriven/nrk-extrapolator-exercise` on CERN GitLab). The NN UT→T integration lives behind `m_use_nn_utt`. RK reference: `device/kalman/ParKalman/include/RungeKuttaExtrapolator.cuh`; extrapUTT: `ParKalmanMethods.cuh:287`. |
| `/data/bfys/gscriven/TE_stack` | **The LHCb stack** — full build env. `build.*/Allen` (built binary), `external/ParamFiles/data/.../params_UTT_v0.tab` (extrapUTT coefficients), `Rec/Tr/TrackExtrapolators` (HLT2 C++ extrapolators). Read-only, not ours. |

## Key git state (deliverable repo)
Last v1 commits: `283b03b` wave-2, `b7e4b19` speed bench, `5f14f8c` A4 rebuild, `dc21ee5` gen-4
three-arm, `dee4cdf` conventions schema. **Working tree clean and push-ready** (cruft removed at
close-out). Note: these v1 commits were not pushed at archive time — push or branch as preferred when
rescoping.

## Headline artifacts
- **NN candidate (locked, weak-field):** `candidate/pinn_v2_ALLEN_v1/` (checkpoint + `PINN_V2_UTT.cuh`).
- **gen-4 corpus:** `<lab>/experiments/gen_3/data/train_10M_gen4.npz` (+ `.meta.json`).
- **wave-2 corpus:** `train_wave2_deploy.npz` (+ schema/meta JSON).
- **external truth check:** the extrapUTT bake-off — C++ `allen_bridge/extraputt_baseline.cpp`
  (build via `allen_bridge/build_extraputt.sh`, `ALLEN_DIR`) and Python port `paper_p0/extraputt_py.py`.
- **plane truth:** `paper_p0/v8r1_plane_truth.npz`, `plane_ref_v8r1.npz`.
- **speed:** `gates/baseline/throughput.json`, `allen_bridge/bench/`.
- **chart:** `<lab>/experiments/flattening/` → carried into `/data/bfys/gscriven/Ex_rep/Chart/`.

## Environment
Python: `/data/bfys/gscriven/conda/envs/TE/bin/python` (torch 2.9 + cu128, cupy 14 for the bench).
HTCondor on Nikhef/stoomboot: GPU jobs need `request_gpus=1` (no `CUDACapability` requirement; the pool
advertises `GPUs_Capability`), small CPU/mem to fit free V100 slots. No GPU on the login node.

## Notion (same project, v1 content archived 06-16)
Project "Track Extrapolation". Databases: To-do, Write Up (for PhDs), Literature & Resources. v1
write-ups/plans/to-dos are marked archived (pointer to this folder). **Literature retained as valid.**
