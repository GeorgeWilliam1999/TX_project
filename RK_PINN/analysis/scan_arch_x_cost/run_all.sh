#!/usr/bin/env bash
# Regenerate every figure + table for the arch x cost scan analysis.
# Reads trained models from TrackExtrapolation/experiments/gen_3/trained_models/scan{A,B}_*
# and the speed bench Ex_rep/RK_PINN/results/scan_speed.json. Writes ./figures and ./results.
set -e
cd "$(dirname "$0")"
echo "[1/3] per-variable accuracy eval (loads 257MB corpus, 12 models) ..."
python3 scan_eval_per_variable.py
echo "[2/3] training-history / convergence plots ..."
python3 scan_plot_histories.py
echo "[3/3] cross-cut summary (accuracy-vs-speed, by-momentum, table) ..."
python3 plot_summary.py
echo "done -> figures/  results/"
