#!/usr/bin/env bash
# run_opt.sh — HTCondor GPU-slot worker for the PINN optimisation bench.
# Re-times the 3 baselines (RK / extrapUTT / PINN) AND the optimised variants on
# the SAME slot, so every number is apples-to-apples on one V100.
set -uo pipefail

REPO=/data/bfys/gscriven/track-extrapolation-pinn
BENCH="$REPO/allen_bridge/bench"
WORK=/data/bfys/gscriven/Ex_rep/RK_PINN
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
SP=/data/bfys/gscriven/conda/envs/TE/lib/python3.10/site-packages

export CUPY_CACHE_DIR="$WORK/.cupy_cache"
export PYTHONPATH="$BENCH/_pyenv:${PYTHONPATH:-}"
for comp in cuda_nvrtc cuda_runtime nvjitlink cuda_cupti; do
  if [ -d "$SP/nvidia/$comp/lib" ]; then
    export LD_LIBRARY_PATH="$SP/nvidia/$comp/lib:${LD_LIBRARY_PATH:-}"
  fi
done
mkdir -p "$CUPY_CACHE_DIR" "$WORK/results"

echo "=========================================="
echo "  PINN optimisation bench"
echo "  host: $(hostname)   date: $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader || echo "no nvidia-smi"

echo; echo "### [1/2] baselines (same-slot re-time) ###"
"$PY" "$BENCH/microbench.py" --warmup 200 --repeats 50 \
    --out "$WORK/results/baselines_sameslot.json" "$@" || echo "baseline rerun FAILED (continuing)"

echo; echo "### [2/2] optimised variants ###"
exec "$PY" "$WORK/bench/microbench_opt.py" --warmup 200 --repeats 50 \
    --out "$WORK/results/tier1_opt.json"
