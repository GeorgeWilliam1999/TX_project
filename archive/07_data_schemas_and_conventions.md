# 07 — Data Schemas & Conventions

## Physics contract (applies to all v1 artifacts after the κ fix)
- **κ:** `kappa = 1e-3 * qop`, `qop = 0.299792458 * q/p[1/GeV]` (= Allen c·q/p).
- **Field:** LHCb FieldMap **v8r1 down** (CVMFS `field.v8r1.down.bin`, the `magfield.bin` Allen loads;
  loader `core/field_v8r1.py`). **Raw sign: MagDown By < 0. No sign flips.**
- **Polarity pairing:** production `extrapUTT` ↔ `m_polarity = -1`.
- **Truth integrator:** vectorised RK4, 5 mm fixed step (validated vs extrapUTT at 15 µm).
- **Legacy (do NOT use for new work):** `core/magnetic_field.py` (toy twodip; weak + sign-flipped via
  `get_field_numpy`); κ = 1e-6.

## Training corpus — `train_10M_gen4.npz` (9,188,440 tracks), keys X, Y, P (all float32)
| array | cols | meaning | units |
|---|---|---|---|
| X[N,7] | 0..6 | x, y, tx, ty, qop, z0, dz | mm, mm, –, –, c·q/p[1/GeV], mm, mm (signed) |
| Y[N,5] | 0..4 | x, y, tx, ty, qop @ z0+dz | mm, mm, –, –, c·q/p (= X[:,4]) |
| P[N] | – | p = 0.299792458/|qop| | GeV |

Ranges: x,y ∈ [-3900,3900]; z0 ∈ [0,14000]; dz ∈ [-10000,10000], |dz|≥25; p ∈ [1,200].
Population: 70 % PV-pointing / 30 % broad. Generation gates (`merge_validate_v2.py`):
G-INT reprop <1e-3 mm · G-PHY long-step median |dtx| ∈ [0.02,0.5] rad · G-POP ≥8M, balanced.

### Appropriateness caveat (why wave-1 failed)
The corpus is *correct* but *mis-weighted* for UT→T: UT→T = 0.145 % of rows; 65 % of steps <1 m;
target spans 9.9 decades (0 µm → 7.5 m). Wave-2's restratified set
(`train_wave2_deploy.npz`): 4.0M general (acceptance-capped x≤3000, y≤2500) + 1.2M UT→T-focused →
UT→T 23.1 %.

## Plane reference — `plane_ref_v8r1.npz`
`X_plane[N,5]` @ z=2665, `Y_true[N,5]` @ z=7826 (v8r1 RK truth). Companion `plane_states_v8r1.csv`
(header `x,y,tx,ty,qop_corpus`) → extrapUTT driver → `plane_poly_v8r1_polm1.csv`.

## A4 Jacobian reference — `For_Allen/artifacts/phase1a/{J_rk4_reference,X_a4}.npy`
`J[N,5,5]` = d(state_out)/d(state_in) from fp64 RK; rebuilt at physical κ on 06-14 (weak-field 05-12
version preserved as `*_weakfield_2026-05-12.npy`). Gate frob_rel < 0.05.

## Checkpoint dir — `trained_models/<name>/`
`best_model.pt` (model_state_dict + config), `config.json`, `normalization.json`, `history.json`
(curves, best_epoch, test_final), `test_indices.npy`.

## Eval output — `*_three_arm.json`
per arm: median/p68/p95/p99 dx (µm), median dtx (µrad), median dx by |q/p| quartile (hi→lo p),
spec-weighted variants. Arrays in the companion `*_arrays.npz`.
