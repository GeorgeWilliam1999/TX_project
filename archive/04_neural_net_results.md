# 04 — Neural-Net Results (gen-1 → gen-4 + wave-2)

## Architecture (PINN_v2, the locked line)
A ζ-gated correction envelope: `u_θ(x0,ζ) = y_base + ζ·δ_θ([x̃0,ζ])`, so the initial condition is exact
by construction; the network learns only a bounded correction. Physics entered the **training loss**
only (a PDE residual via forward-mode JVP), keeping inference field-free (satisfies the replacement
criterion). Locked candidate `pinn_v2_ALLEN_v1`: 10,372 params, 40 kB fp32 (fits the 64 kB budget).
Variants explored: a **kick-scaled head** (correction × κ·Δz, q/p-exact by construction) and λ_pde / λ_ic
ablations.

## Weak-field era (gen-3) — the misleading "success"
Headline ~12 µm median over the full signed-Δz distribution; ~293 µm on UT→T; passed the A4 Jacobian
gate. **All of this was on the ×1000-weak toy field** — i.e. a near-field-free, trivially easy problem.

## Physical era (gen-4) — the honest numbers

### Three-arm bake-off (06-14), real v8r1 plane, 7,947 PV-pointing tracks, z 2665→7826

| arm | median |Δx| | low-p quartile |
|---|---|---|
| **extrapUTT (incumbent)** | **10.9 µm** | 475 µm |
| best gen-4 NN (wave-1, λ=0) | 175 mm | 812 mm |
| straight line (no model) | 225 mm | 1.01 m |

The wave-1 NN removed only ~22 % of the bend — barely better than doing nothing.

### λ ablation flipped at physical κ
Weak-field: λ=0 beat λ=0.1 by 3.4–5.6×. Physical field: on the full mixed distribution λ=0.1 wins
(test median 1.3 vs 4.4 mm at 2M); on the hard UT→T plane λ=0 wins (the physics-regularised model
collapses toward the straight-line prior where the bend is largest). Regime-dependent; no clean win.

### Wave-2 — the proper retry (the verdict)
Diagnosed root causes from a data audit: **UT→T was 0.145 % of the corpus**, 65 % of steps <1 m, and
the target spanned **9.9 decades** (0 µm → 7.5 m). Fixes applied: restratified corpus (UT→T → 23 %,
acceptance-capped, 5.2M tracks), residual/kick parametrisation, range-aware loss, proper schedule
(best_epoch ~90, not 3–7), and a **capacity sweep h32 → h384**.

| width | UT→T median | bulk median |
|---|---|---|
| h32 | 3832 µm | 1.09 mm |
| h64 | 3300 µm | 1.01 mm |
| h96 | 3196 µm | 1.00 mm |
| h128 | 3162 µm | 0.99 mm |
| h256 | 3109 µm | 1.00 mm |
| h384 | 3130 µm | 0.97 mm |

The data fix improved UT→T **175 mm → ~3.1 mm (≈55×)** — but it **plateaus at ~3 mm regardless of
width or λ.** That is a genuine floor, **not** a capacity ceiling — and still **~285× worse than
extrapUTT's 11 µm.**

## Verdict
**The NN accuracy route is closed.** A compact, field-free network — even with correct data, the right
parametrisation, proper training, and 4× the width — cannot approach the production polynomial on the
real field. Combined with the speed result (`05_speed_benchmark.md`, the NN is also *slower* than both
RK and the polynomial), the NN replacement has no surviving value axis for UT→T.
