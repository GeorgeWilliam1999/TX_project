#!/usr/bin/env bash
set -uo pipefail
BENCH=/data/bfys/gscriven/track-extrapolation-pinn/allen_bridge/bench
WORK=/data/bfys/gscriven/Ex_rep/RK_PINN
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
SP=/data/bfys/gscriven/conda/envs/TE/lib/python3.10/site-packages
export CUPY_CACHE_DIR="$WORK/speed/.cupy_cache"
export PYTHONPATH="$BENCH/_pyenv:${PYTHONPATH:-}"
for c in cuda_nvrtc cuda_runtime nvjitlink cuda_cupti; do [ -d "$SP/nvidia/$c/lib" ] && export LD_LIBRARY_PATH="$SP/nvidia/$c/lib:${LD_LIBRARY_PATH:-}"; done
mkdir -p "$CUPY_CACHE_DIR"
echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
exec "$PY" "$WORK/speed/time_nystrom.py"
