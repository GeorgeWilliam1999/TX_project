# 09 — Verdict & Lessons (and open questions for the rescope)

## The verdict
For the **UT→T step on the real LHCb field**, neither v1 surrogate is competitive with the production
`extrapUTT` polynomial:

| candidate | accuracy (median UT→T) | speed vs RK | footprint |
|---|---|---|---|
| **extrapUTT (incumbent)** | **15 µm** | 2.4× faster | 0.96 MB |
| neural net (wave-2, best, tuned) | ~3 mm (~285× worse) | 1.2× slower | 40 kB |
| analytic chart (v8r1, best) | ~3.9 mm (~260× worse) | fast | 63 kB |
| straight line | 237 mm | — | 0 |

Both routes win only on **footprint**, which does not justify a slower, far-less-accurate surrogate.
The micron-level v1 "wins" were artifacts of the ×1000-weak field. **v1 as scoped (beat RK/extrapUTT
at UT→T) is closed.**

## What is genuinely reusable (do not rebuild)
1. **The external truth check** — the extrapUTT bake-off (C++ + faithful Python port, both validated).
   This is the single most valuable asset; it is what makes results trustworthy.
2. **Correct physics + data machinery** — κ=1e-3, v8r1 loader, polarity, RK4 truth generator, the
   validation gates, the locked data schemas.
3. **The benchmarking harnesses** — three-arm accuracy eval, the GPU/CPU speed harness, the A4 gate.
4. **The flattening principle** — sound on smooth fields; needs a localised field representation.
5. **The methods/cautionary narrative** — the κ discovery (self-consistency ≠ correctness) is a strong
   standalone contribution.

## Honest open questions for the rescope (start from Allen)
1. **What does Allen actually need that isn't already solved?** UT→T is well-served by extrapUTT. The
   *general* extrapolation (arbitrary dz, where no polynomial exists, and RK is the only option) is the
   un-attacked opportunity — is RK there a real bottleneck, and is a surrogate competitive *there*?
2. **Is the value axis accuracy, speed, or footprint?** v1 implicitly chased accuracy at UT→T and lost.
   A rescope should pick the axis where Allen has a genuine, measured pain point first.
3. **Could a localised chart match extrapUTT at a footprint win?** Possibly (feasibility probe suggests
   ~0.1–1 mm), but parity with 15 µm is unproven and it converges toward extrapUTT's design.
4. **Should the contribution be the method or the methodology?** The cautionary-tale paper is real and
   defensible regardless of whether any surrogate ever wins.

## Recommended posture for v2
Re-scope **end-to-end from Allen**: profile where extrapolation actually costs Allen, identify the
specific step(s) with no good incumbent, define the value axis and the bar **before** building, and
keep the extrapUTT bake-off as the non-negotiable acceptance gate from day one.
