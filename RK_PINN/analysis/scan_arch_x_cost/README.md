# Architecture × cost-function scan — analysis

Analysis of the 2026-06-23 scan that tested whether any architecture or loss tweak
pushes the neural track extrapolator below its **~3 mm UT→T accuracy floor**.
Analysed 2026-06-25. Regenerate everything with `./run_all.sh`.

## The runs (12, Condor cluster 4860628)
- **Block A — architecture** (`scanA_*`, width 64, `residual_rel` loss): depth ∈ {2, 3} × activation ∈ {tanh, silu, gelu, **sin/SIREN**}. 8 runs.
- **Block B — cost function** (`scanB_*`, [96,96] tanh): `alpha=0.5` blend, Huber `delta=2`, inverse-p weighting, `log_cosh`. 4 runs. Baseline = `wave2_resid_h96`.
- **Speed exam** (cluster 4860629) → `../../results/scan_speed.json` (NVIDIA L40S, cc89).

## Headline findings
1. **The ~3 mm floor holds.** Best test UT→T median = **d3_tanh 3.03 mm**; the usable spread is **3.0–4.2 mm**. Nothing approaches the 1 mm target, let alone extrapUTT (~15 µm).
2. **It is a low-momentum, in-bending-plane floor.** Error scales **~40×** from high-p (~0.2 mm) to low-p (~7–8 mm); it lives almost entirely in x/tx (median |Δty| ≈ 0.06 mrad). See `fig_accuracy_by_momentum` and `fig_scan_acc_vs_variable_*`.
3. **Architecture is irrelevant** — all 8 Block-A curves overlap to within ~10% at every momentum ⇒ the limit is representation/labels, not capacity/activation. SIREN did not move the floor (and was unstable at depth 3: `scanA_d3_sin` early-stopped @62).
4. **No cost function rescues it.** `inv_p` (aimed at low-p) made low-p slightly *worse*; `log_cosh` worst; `delta2` ≈ baseline. `scanB_alpha05` is a bias/variance pathology — best typical accuracy (overall median 769 µm) but a **986 mm p95**, so its UT→T-subset median is 254 mm (fits the easy majority, abandons the hard long extrapolations).
5. **Speed (L40S):** depth costs 2–3× for ~5–10% accuracy (d2 ~0.32–0.52 ns vs d3 ~0.86–1.12 ns). ⚠️ These are L40S; the 4.85/0.91 ns and extrapUTT/RK anchors were V100 — **not directly comparable**.

**Verdict:** pure-NN accuracy route is closed → pivot to **residual-to-extrapUTT** (learn the small correction on the incumbent, low-p focused).

## Figures (`figures/`)
| file | what it shows |
|---|---|
| `fig_accuracy_vs_qop.png` | position accuracy vs **signed q/p** (the model's own input var); valley at q/p≈0 (high-p), rising to the low-p bend tail; top axis labels p in GeV |
| `fig_accuracy_by_momentum.png` | median \|Δpos\| per momentum quartile, every run (the key plot) |
| `fig_accuracy_vs_speed.png` | test UT→T median vs kernel speed; filled=real weights, open=placeholder/arch-proxy |
| `fig_scan_acc_vs_variable_blockA.png` | accuracy vs p / \|position\| / tx / ty — depth×activation |
| `fig_scan_acc_vs_variable_blockB.png` | same axes — cost-function variants |
| `fig_scan_acc_channels_blockA.png` | \|Δtx\|, \|Δty\| vs p — depth×activation |
| `fig_scan_acc_channels_blockB.png` | same — cost-function variants |
| `scan_hist_blockA.png` | convergence: val UT→T median + train loss vs epoch — depth×activation |
| `scan_hist_blockB.png` | same — cost-function variants |

## Tables (`results/`)
- `scan_summary_table.{json,csv}` — one row/run: arch, loss, convergence, test UT→T median + p95, speed, train time.
- `scan_per_variable.json` — overall + by-p-quartile medians + slope channels.
- `scan_histories_summary.json` — best/test UT→T median, best epoch, train time.

## Provenance
- Models: `TrackExtrapolation/experiments/gen_3/trained_models/scan{A,B}_*` (trained 2026-06-23, cluster 4860628).
- Data: `…/gen_3/data/train_wave2_deploy.npz` (gen-4 physical-κ corpus); test split = `test_indices.npy` (seed 42, **519,673** tracks, shared across runs).
- Speed: `Ex_rep/RK_PINN/results/scan_speed.json` (cluster 4860629, NVIDIA L40S cc89).
- Model code: `track-extrapolation-pinn` @ `283b03b` (`models/architectures.py`). Analysis repo `Ex_rep` @ `c5eb558`.
- Metric: `utt_median_dx_um` = UT→T-subset median |Δx| (the one that matters). NB the per-track `median_dx_mm` ≈ 1 mm field is the *all-pairs* median and is misleading.
