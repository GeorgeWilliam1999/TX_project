# Track Extrapolation — Archive (v1, pre-rescope)

**Archived:** 2026-06-16 · **Status:** v1 closed. The project is being re-scoped end-to-end from Allen.
This folder is the durable record of everything done in v1. Notion content for v1 is archived; the
literature remains valid. The chart code (the one direction worth carrying forward) lives in
`../Chart/`.

## The project in one paragraph
LHCb reconstruction repeatedly asks: *given a charged track's state at one z-plane, where is it at
another plane after curving through the dipole magnet?* That is **track extrapolation**. The
production method is an adaptive **Runge–Kutta (RK)** integrator — accurate, but it reads a 3-D field
map and is called O(10⁶)×/event. v1's goal was to **replace it with a faster surrogate that needs no
field map at inference** — either a small **neural network (NN)** or an **analytic "chart."**

## The bottom line (why we're restarting)
Both surrogates **lose to the incumbent on the real LHCb field, on both accuracy and speed:**

| candidate (UT→T step, real v8r1 field) | median error | speed vs RK | verdict |
|---|---|---|---|
| **extrapUTT** (production polynomial) | **14.9 µm** | 2.4× faster | the bar |
| RK (current general method) | (truth) | 1× | what we meant to replace |
| neural net (best, wave-2, tuned) | ~3 mm | **1.2× slower** | dead — accuracy floor + slower |
| analytic chart (best, v8r1) | ~3.9 mm | (fast) | loses 260× on real field |
| straight line (no model) | 237 mm | — | floor |

The micron-level v1 results that looked like success were an **artifact of a physics bug**: every
dataset had the magnet **1000× too weak** (the κ bug), making the problem almost field-free and
trivially easy. On the real field, neither surrogate is competitive with the polynomial that already
exists. See `09_verdict_and_lessons.md`.

## What is NOT wasted
- The **κ / field discovery** is a strong methods/cautionary result (a 1000× bug hid through three
  generations of internally-consistent metrics; one external baseline caught it).
- The **honest, externally-validated benchmarking machinery** (the extrapUTT bake-off, the speed
  harness, the data schemas, the conventions) carries straight into the rescope.
- The **chart's flattening principle** is sound on smooth fields; a *localised* redesign might yet
  close the gap (see `06_chart_programme.md` and `../Chart/`).

## Read in this order
1. `01_project_and_goal.md` — the goal, the Allen context, the "replacement criterion."
2. `02_timeline.md` — the full chronological arc.
3. `03_kappa_and_field_discovery.md` — **the central finding.**
4. `04_neural_net_results.md` — the NN line, gen-1→4 + wave-2, the accuracy floor.
5. `05_speed_benchmark.md` — throughput; the NN is slower than RK and the polynomial.
6. `06_chart_programme.md` — the analytic chart, and why it also loses on the real field.
7. `07_data_schemas_and_conventions.md` — corpus schema, κ/field/polarity, validation gates.
8. `08_repos_and_artifacts.md` — where every code/data artifact lives.
9. `09_verdict_and_lessons.md` — honest conclusions + open questions for the rescope.

## Glossary (decoder ring)
- **RK** — Runge–Kutta integrator; the current method we tried to replace.
- **extrapUTT** — a production polynomial shortcut for the UT→T step; our accuracy/speed yardstick.
- **UT→T** — the hardest step, from the UT tracker (before the magnet) to the T stations (after).
- **NN / PINN_v2** — our trainable neural surrogate.
- **chart / flattening** — the analytic (table-based) surrogate; see `../Chart/`.
- **κ (kappa)** — the magnet-bending constant; was 1000× too weak in all v1 data (the big bug).
- **gen-1/2/3** — NN rounds on the buggy weak-field data. **gen-4** — the corrected dataset.
- **wave-1/2** — two NN training rounds on gen-4 (quick try / serious tuned try).
- **A4** — the Jacobian quality gate (sensitivities, needed by the Kalman filter).
- **Allen** — LHCb's GPU reconstruction software; deployment target, kept pristine.
