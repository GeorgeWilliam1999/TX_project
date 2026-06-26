# bench/ — the measurement harness

Reproduces every number in `../RESULTS.md` and the figures in `../../throughput/`.
Each `condor/opt*.sub` lands one exclusive V100 slot, NVRTC-compiles the kernels against the
locked headers, and writes a results JSON. The deployable kernel itself lives in
`../kernel/pinn_v2_utt_fast.cuh`; this folder is only how it was measured.

| kernel file | driver | submit | result | what it establishes |
|---|---|---|---|---|
| `pinn_opt_kernels.cu` | `microbench_opt.py` | `opt.sub` | `results/tier1_opt.json` + `baselines_sameslot.json` | v1: fused (4.85, bit-exact) beats RK; warp-coop loses; same-slot baselines |
| `pinn_opt_kernels_v2.cu` | `microbench_opt_v2.py` | `opt_v2.sub` | `results/tier1_opt_v2.json` | v2: bit-parity fused; h64 (2.27); ftanh/fp16/launch-bounds dead ends; **cuBLAS GEMM ceiling** (mirage) |
| `pinn_opt_kernels_v3.cu` | `microbench_opt_v3.py` | `opt_v3.sub` | `results/tier1_opt_v3.json` | v3: full-unroll + ILP; **`pinn_h64_fu` 0.91 ns @ 73.8% peak** (fastest); block sweep |

`results/combined.json` bundles all of the above (the single file the figure script reads).

Run any job: `condor_submit ../condor/opt_v3.sub` (etc.). Toolchain: conda env `TE` (cupy + NVRTC,
no external nvcc) on a `Tesla V100-PCIE-32GB` slot; the locked headers + the 1 M-track inputs are read
from `/data/bfys/gscriven/track-extrapolation-pinn` (read-only).
