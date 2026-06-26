# NOTE — read before using these RK4 curves

`rk4_bar.py` here uses the **archived/deprecated `NeuralRK4`** integrator
(`disable_correction=True`). On the gen-4 corpus it **converges to ~2.1 mm off
the truth** — median is identical at n = 8 → 512 (2108 / 2110 / 2112 / 2111 µm),
with an ~2 m p95 and ~11 mm low-p. Converged-but-wrong ⇒ it does **not** reproduce
the gen-4 truth generator (`datagen/generate_data_v2.py:rk4`), despite a matching
κ = 1e-3 prefactor. **These curves are an artifact of a stale integrator
convention, NOT the field-access accuracy bar.**

**The real field-access bar** is the Verified incumbent profile: `extrapUTT` vs the
gen-4 RK4 truth = **~15 µm median, 748 µm low-p, 11.7 mm p99** (Physical-κ Baselines
/ incumbent-profile write-up). The truth integrator is exact by construction (it
defines Y), so it isn't a separate "bar" to recompute.

**Implication for E5.** The quick "retrain the archived `neural_rk4`" path is a dead
end on gen-4: the learned-hybrid jobs `e5_hybrid_n{2,4}` went NaN (a 2–4-step RK4
over the ~5 m UT→T crossing diverges before the learned correction matters), and the
classical base is ~2 mm off truth anyway. A genuine learned-RHS hybrid would have to
be **rebuilt on datagen's truth RHS**. Deprioritised — the bar is known and E1 (pure
field-free multi-step) is the promising lead.
