# F4a — extrapUTT reimplemented & validated; bake-off root cause CORRECTED (2026-06-14)

## Outcome: SANE incumbent reproduced
A faithful, vectorised Python port of Allen's `extrapUTT` now exists:
`experiments/gen_3/paper_p0/extraputt_py.py` (parser + META + `compute_state` + eval),
driver `validate_extraputt.py`. It is **bit-faithful** to the C++: it reproduces the
pre-computed `kappa_val_poly_pol{m1,p1}.csv` to ~1e-3 µm (e.g. polm1 472984.3 vs 472984.4;
polp1 13564.2 vs 13564.4), so the port is correct, not a re-derivation.

**Canonical sane incumbent number** (PV-pointing tracks, real **v8r1** field truth,
qOp = corpus_qop/299.792458, **m_polarity = −1**):

| candidate | median dx | p95 dx | p99 dx | median dy | median dtx |
|---|---|---|---|---|---|
| straight line | 237,109 µm | 1.37 m | 1.77 m | 4643 µm | 90,468 µrad |
| **extrapUTT (pol −1)** | **14.9 µm** | 2178 µm | 11.7 mm | 3.1 µm | 5.7 µrad |

extrapUTT reduces the real 237 mm UT→T bend to a 15 µm residual — exactly what a
production analytic kick map should do. (pol +1 vs v8r1 = 473 mm garbage → the documented
MagDown −1 is unambiguously correct once the field matches.)

## The 369 mm "broken baseline" was NOT a harness wiring bug
The F4 status note (2026-06-11) hypothesised a structural wiring error in the CSV
generator. That is **wrong**: the clean Python port reproduces those CSVs exactly. The
369 mm median is the *genuine* output of a correct extrapUTT eval — because it was run on
the wrong field and the wrong population. Three compounding issues, in order of size:

1. **Field mismatch (dominant).** The whole corpus (`train_10M_gen3`, `utt_plane_ref`) is
   RK truth on the **toy `twodip.rtf`** field. extrapUTT's 25v0 coefficients were fit to
   the **real LHCb v8r1** field. On identical PV-pointing inputs the two fields' UT→T
   endpoints differ by **median 473 mm** (max 4.3 m). extrapUTT predicts the real-field
   trajectory, so it can never match toy-field truth — vs twodip it scores 13.6 mm (pol+1)
   / 473 mm (pol−1); vs v8r1 it scores **14.9 µm** (pol−1).
2. **Population mismatch.** extrapUTT is a local expansion around the **PV-pointing
   manifold**: the grid bin comes from position (`xx=x/(zi·Txmax)`, valid only for
   |x|≲zi·Txmax=799 mm) and `ux=(tx−x/zi−bendx·qop)·200`, `uy=...` are the *small*
   deviations from pointing. The uniform-phase-space corpus has x⊥tx
   (median |x−tx·zi| = 1749 mm), so only 6.3 % of `utt_plane_ref` is even in-grid and
   `ux,uy` are O(10–60) — far outside the expansion. Result: garbage even ignoring field.
3. **Polarity sign.** The headline `polm1` CSV used m_polarity=−1 *against the toy field*,
   where −1 is the wrong sign (toy-field cancellation favours +1). Against the correct
   field, −1 is right. This was a red herring created by issue 1.

## Implication for F4b (the bake-off) — needs a common field + population
The chart (12.1 µm) and extrapUTT (14.9 µm) currently live in **different field worlds**:
- chart tables (`chart_tables.npz`) are built from the toy twodip field → validated vs
  twodip truth on the uniform corpus pool;
- extrapUTT is fit to v8r1 → validated vs v8r1 truth on PV-pointing tracks.

A fair head-to-head requires one field and one population. Since the chart is meant to
**replace extrapUTT in Allen (which runs on v8r1)**, the deployment-relevant comparison is
**real v8r1 field, PV-pointing (production-like) population**. That means: rebuild the chart
field-integral/multipole tables from v8r1, regenerate a v8r1 truth pool, then score both.
The toy-field 12.1 µm is the gen-1→3 corpus number, not the deployment number.

## Reproduce
```
cd experiments/gen_3/paper_p0
python extraputt_py.py        # parse + print META / coeff spot-check
python validate_extraputt.py  # sweep qop-scale x polarity vs utt_plane_ref
# v8r1 PV-pointing comparison: load v8r1_plane_truth.npz, extrapUTT(p,X,polarity=-1)
```

## Next
- F4b: decide the common field+population for the bake-off (recommend v8r1 + PV-pointing),
  rebuild chart tables on v8r1, publish the comparison table, run chart through A4 gate.
- F4a is otherwise CLOSED: extrapUTT is reproduced and the incumbent number is sane.
- Derivatives (der_tx/der_ty/der_qop) not yet ported — small add for the A4 Jacobian gate.
