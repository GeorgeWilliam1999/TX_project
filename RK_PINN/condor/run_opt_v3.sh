#!/usr/bin/env bash
set -uo pipefail
REPO=/data/bfys/gscriven/track-extrapolation-pinn
BENCH="$REPO/allen_bridge/bench"
WORK=/data/bfys/gscriven/Ex_rep/RK_PINN
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
SP=/data/bfys/gscriven/conda/envs/TE/lib/python3.10/site-packages
export CUPY_CACHE_DIR="$WORK/.cupy_cache"
export PYTHONPATH="$BENCH/_pyenv:${PYTHONPATH:-}"
for comp in cuda_nvrtc cuda_runtime nvjitlink cuda_cupti cublas; do
  [ -d "$SP/nvidia/$comp/lib" ] && export LD_LIBRARY_PATH="$SP/nvidia/$comp/lib:${LD_LIBRARY_PATH:-}"
done
mkdir -p "$CUPY_CACHE_DIR" "$WORK/results"
echo "host: $(hostname)  date: $(date)"
exec "$PY" "$WORK/bench/microbench_opt_v3.py" --warmup 200 --repeats 50 --out "$WORK/results/tier1_opt_v3.json"
