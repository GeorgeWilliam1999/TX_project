#!/usr/bin/env bash
# Worker for the scan speed exam (V100). Regenerates kernels (picks up any real weights
# now available) then times them. Conda TE for torch (weight export) + cupy in _pyenv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/data/bfys/gscriven/conda/envs/TE/bin/python3
echo "=== scan speed exam: host=$(hostname) $(date) ==="
"$PY" "$HERE/gen_fused_kernels.py"
"$PY" "$HERE/time_scan_speed.py"
