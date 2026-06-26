#!/usr/bin/env python3
"""scan_plot_histories.py — training histories for the 2026-06-23 arch x cost scan.

Reads each run's history.json (val carries utt_median_dx_um, median_pos_mm, ...;
train carries data_loss/loss/lr). Produces:
  figures/scan_hist_blockA.png   depth x activation (utt-median + train-loss vs epoch)
  figures/scan_hist_blockB.png   cost-function variants (same)
and a results/scan_histories_summary.json table (best utt-median, epoch, final test).
Skips runs not present yet, so it can be dry-run while the scan is still training.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/trained_models"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

# (display label, run dir).  Block A baseline = wave2_resid_h64 (same cfg as d2_tanh).
BLOCK_A = [
    ("d2 tanh", "scanA_d2_tanh"), ("d2 silu", "scanA_d2_silu"),
    ("d2 gelu", "scanA_d2_gelu"), ("d2 sin",  "scanA_d2_sin"),
    ("d3 tanh", "scanA_d3_tanh"), ("d3 silu", "scanA_d3_silu"),
    ("d3 gelu", "scanA_d3_gelu"), ("d3 sin",  "scanA_d3_sin"),
]
BLOCK_B = [
    ("h96 baseline (resid)", "wave2_resid_h96"), ("alpha=0.5 blend", "scanB_alpha05"),
    ("huber delta=2", "scanB_delta2"), ("inv-p weight", "scanB_invp"),
    ("log-cosh", "scanB_logcosh"),
]


def load(run):
    p = os.path.join(LAB, run, "history.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def plot_block(block, title, fname):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(block)))
    summary = {}
    for (lab, run), c in zip(block, cmap):
        H = load(run)
        if H is None:
            print(f"  [skip] {run} (no history yet)"); continue
        val = H["val"]; tr = H["train"]
        ep = np.arange(1, len(val) + 1)
        utt = np.array([v.get("utt_median_dx_um", np.nan) for v in val], float)
        dloss = np.array([t.get("data_loss", np.nan) for t in tr], float)
        axL.plot(ep, utt, "-", color=c, lw=1.6, label=lab)
        axR.plot(np.arange(1, len(dloss) + 1), dloss, "-", color=c, lw=1.4, label=lab)
        best = H.get("best_select_metric")
        tf = H.get("test_final", {})
        summary[run] = {
            "label": lab,
            "best_utt_median_um": float(np.nanmin(utt)) if np.isfinite(utt).any() else None,
            "best_epoch": H.get("best_epoch"),
            "test_utt_median_um": tf.get("utt_median_dx_um"),
            "test_median_pos_mm": tf.get("median_pos_mm"),
            "test_median_dtx_mrad": tf.get("median_dtx_mrad"),
            "test_median_dty_mrad": tf.get("median_dty_mrad"),
            "train_time_min": H.get("training_time_s", 0) / 60.0,
            "params": H.get("test_final", {}).get("params"),
        }
    axL.set_yscale("log"); axL.axhline(3000, color="k", ls="--", lw=.7)
    axL.text(2, 3300, "3 mm floor", fontsize=8)
    axL.axhline(1000, color="g", ls=":", lw=.7); axL.text(2, 1050, "1 mm target", fontsize=8, color="g")
    axL.set_xlabel("epoch"); axL.set_ylabel("val UT→T median |Δx| [µm]")
    axL.set_title("selection metric (val)"); axL.legend(fontsize=8, ncol=2); axL.grid(alpha=.3, which="both")
    axR.set_yscale("log"); axR.set_xlabel("epoch"); axR.set_ylabel("train data loss")
    axR.set_title("training loss"); axR.legend(fontsize=8, ncol=2); axR.grid(alpha=.3, which="both")
    fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, fname); fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)
    return summary


def main():
    s = {}
    print("Block A (depth x activation):")
    s.update(plot_block(BLOCK_A, "Scan Block A — depth × activation (width 64, residual_rel loss)", "scan_hist_blockA.png"))
    print("Block B (cost function):")
    s.update(plot_block(BLOCK_B, "Scan Block B — cost function ([96,96] tanh)", "scan_hist_blockB.png"))
    json.dump(s, open(os.path.join(RES, "scan_histories_summary.json"), "w"), indent=2)
    print("\nwrote results/scan_histories_summary.json")
    print(f"\n{'run':<20} {'best_utt_um':>12} {'test_utt_um':>12} {'test_pos_mm':>12} {'min':>6}")
    for run, d in s.items():
        b = d["best_utt_median_um"]; t = d["test_utt_median_um"]; pm = d["test_median_pos_mm"]
        print(f"{run:<20} {b if b is None else round(b,1)!s:>12} "
              f"{t if t is None else round(t,1)!s:>12} {pm if pm is None else round(pm,3)!s:>12} "
              f"{round(d['train_time_min'],1)!s:>6}")


if __name__ == "__main__":
    main()
