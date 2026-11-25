#!/usr/bin/env python3
"""Late-τ log-scale figures (10–200); read-only merged_tau_summary.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations/cross_injection_followup_20261125"
LATE_CSV = Path(
    "/yunity/jhwlds/experiments/fmnist_split_cross_injection/"
    "analysis/late_tau_extension_20261124/merged_tau_summary.csv"
)
LATE_TAUS = [10, 25, 50, 100, 150, 200]
COL_AA, COL_AB = "#1f77b4", "#ff7f0e"
COL_GAP_N, COL_GAP_O = "#2ca02c", "#9467bd"


def load_late_band() -> list[dict]:
    rows = list(csv.DictReader(LATE_CSV.open(encoding="utf-8")))
    by_tau = {int(r["tau"]): r for r in rows}
    missing = [t for t in LATE_TAUS if t not in by_tau]
    if missing:
        raise KeyError(f"Missing τ in CSV: {missing}")
    return [by_tau[t] for t in LATE_TAUS]


def style_log_tau_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(10, 200)
    ax.set_xticks(LATE_TAUS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Injection τ (log scale)")
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.25, linestyle=":")


def annotate_tau(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    for xi, yi in zip(x, y, strict=True):
        ax.annotate(
            f"{int(xi)}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color="#444444",
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_late_band()
    x = np.array(LATE_TAUS, dtype=float)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 200})

    aa = np.array([float(r["A_to_A_one_rank_ndcg10_mean"]) for r in data])
    ab = np.array([float(r["A_to_B_one_rank_ndcg10_mean"]) for r in data])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, aa, "o-", color=COL_AA, label="A→A One-Step", ms=7, lw=2)
    ax.plot(x, ab, "s-", color=COL_AB, label="A→B One-Step", ms=7, lw=2)
    style_log_tau_axis(ax)
    ax.set_ylabel("Mean rank NDCG@10 (projection set average)")
    ax.set_ylim(0.68, 1.02)
    ax.set_title("Figure A — Late-τ One-Step NDCG@10 (log τ, 10–200)")
    annotate_tau(ax, x, aa)
    annotate_tau(ax, x, ab)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path_a = OUT / "late_log_figureA_ndcg10.png"
    fig.savefig(path_a, bbox_inches="tight")
    plt.close(fig)

    aa_o = np.array([float(r["A_to_A_one_top10_overlap_mean"]) for r in data])
    ab_o = np.array([float(r["A_to_B_one_top10_overlap_mean"]) for r in data])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, aa_o, "o-", color=COL_AA, label="A→A One-Step", ms=7, lw=2)
    ax.plot(x, ab_o, "s-", color=COL_AB, label="A→B One-Step", ms=7, lw=2)
    style_log_tau_axis(ax)
    ax.set_ylabel("Mean top-10 overlap (projection set average)")
    ax.set_ylim(0.60, 1.02)
    ax.set_title("Figure B — Late-τ One-Step top-10 overlap (log τ, 10–200)")
    annotate_tau(ax, x, aa_o)
    annotate_tau(ax, x, ab_o)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path_b = OUT / "late_log_figureB_top10_overlap.png"
    fig.savefig(path_b, bbox_inches="tight")
    plt.close(fig)

    g_n = np.array([float(r["gap_one_rank_ndcg10"]) for r in data])
    g_o = np.array([float(r["gap_one_top10_overlap"]) for r in data])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    for ax, series, color, title, ylab in [
        (axes[0], g_n, COL_GAP_N, "NDCG gap (A→A − A→B)", "NDCG gap"),
        (axes[1], g_o, COL_GAP_O, "Top-10 overlap gap (A→A − A→B)", "Overlap gap"),
    ]:
        ax.plot(x, series, "o-", color=color, ms=7, lw=2)
        style_log_tau_axis(ax)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, loc="left")
        annotate_tau(ax, x, series)
        ymax = float(series.max()) * 1.08
        ymin = float(series.min()) * 0.92
        ax.set_ylim(ymin, ymax)
    axes[0].set_xlabel("")
    fig.suptitle("Figure C — Late-τ One-Step preservation gaps (log τ, 10–200)", y=1.01)
    fig.tight_layout()
    path_c = OUT / "late_log_figureC_gaps.png"
    fig.savefig(path_c, bbox_inches="tight")
    plt.close(fig)

    verification = {
        "source_csv": str(LATE_CSV),
        "taus": LATE_TAUS,
        "A_to_A_one_rank_ndcg10_mean": aa.tolist(),
        "A_to_B_one_rank_ndcg10_mean": ab.tolist(),
        "A_to_A_one_top10_overlap_mean": aa_o.tolist(),
        "A_to_B_one_top10_overlap_mean": ab_o.tolist(),
        "gap_one_rank_ndcg10": g_n.tolist(),
        "gap_one_top10_overlap": g_o.tolist(),
        "outputs": [str(path_a), str(path_b), str(path_c)],
    }
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()