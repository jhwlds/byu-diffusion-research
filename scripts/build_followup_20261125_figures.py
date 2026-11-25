#!/usr/bin/env python3
"""Figures for 2026-11-25 follow-up report (CPU, read-only CSV/JSON)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations/cross_injection_followup_20261125"
LATE = Path("/yunity/jhwlds/experiments/fmnist_split_cross_injection/analysis/late_tau_extension_20261124")
CANON = Path("/yunity/jhwlds/experiments/fmnist_split_cross_injection/full_ckpt46000_ab_full_baseline_canonical_20261125")
TAUS = [10, 25, 50, 100, 150, 200, 400, 600, 800]
COL_AA, COL_AB = "#1f77b4", "#ff7f0e"
COL_GAP_N, COL_GAP_O = "#2ca02c", "#9467bd"


def load_late_summary() -> list[dict]:
    rows = list(csv.DictReader((LATE / "merged_tau_summary.csv").open(encoding="utf-8")))
    by_tau = {int(r["tau"]): r for r in rows}
    return [by_tau[t] for t in TAUS]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_late_summary()
    x = np.arange(len(TAUS))
    labels = [str(t) for t in TAUS]
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 200})

    aa = [float(r["A_to_A_one_rank_ndcg10_mean"]) for r in data]
    ab = [float(r["A_to_B_one_rank_ndcg10_mean"]) for r in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, aa, "o-", color=COL_AA, label="A→A One-Step", ms=6)
    ax.plot(x, ab, "s-", color=COL_AB, label="A→B One-Step", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Mean rank NDCG@10 (projection set average)")
    ax.set_ylim(0.65, 1.02)
    ax.set_title("Figure 1 — One-Step NDCG@10 vs injection τ")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "figure1_one_step_ndcg.png", bbox_inches="tight")
    plt.close(fig)

    aa_o = [float(r["A_to_A_one_top10_overlap_mean"]) for r in data]
    ab_o = [float(r["A_to_B_one_top10_overlap_mean"]) for r in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, aa_o, "o-", color=COL_AA, label="A→A One-Step", ms=6)
    ax.plot(x, ab_o, "s-", color=COL_AB, label="A→B One-Step", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Mean top-10 overlap")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("Figure 2 — One-Step top-10 overlap vs injection τ")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "figure2_one_step_overlap.png", bbox_inches="tight")
    plt.close(fig)

    g_n = [float(r["gap_one_rank_ndcg10"]) for r in data]
    g_o = [float(r["gap_one_top10_overlap"]) for r in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, g_n, "o-", color=COL_GAP_N, label="NDCG gap (A→A − A→B)", ms=6)
    ax2 = ax.twinx()
    ax2.plot(x, g_o, "s--", color=COL_GAP_O, label="Overlap gap", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("NDCG gap")
    ax2.set_ylabel("Overlap gap")
    ax.set_title("Figure 3 — One-Step preservation gap vs τ")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "figure3_one_step_gap.png", bbox_inches="tight")
    plt.close(fig)

  # Full denoising auxiliary
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    aa_f = [float(r["A_to_A_full_rank_ndcg10_mean"]) for r in data]
    ab_f = [float(r["A_to_B_full_rank_ndcg10_mean"]) for r in data]
    ax.plot(x, aa_f, "o-", color=COL_AA, label="A→A Full", ms=6)
    ax.plot(x, ab_f, "s-", color=COL_AB, label="A→B Full", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Mean rank NDCG@10 (Full, t=0)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Supplementary — Full denoising NDCG@10")
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "figure_supp_full_ndcg.png", bbox_inches="tight")
    plt.close(fig)

    merged = json.loads((CANON / "analysis/canonical_vs_legacy.json").read_text(encoding="utf-8"))
    c0 = merged["canonical"]["sets"]["0"]
    c1 = merged["canonical"]["sets"]["1"]
    cm = merged["canonical"]["mean"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, metric, val0, val1, vmean, rnd in [
        (axes[0], "Rank NDCG@10", c0["rank_ndcg10"], c1["rank_ndcg10"], cm["rank_ndcg10"], cm.get("random_ndcg10", c0["random_ndcg10"])),
        (axes[1], "Top-10 overlap", c0["top10_overlap"], c1["top10_overlap"], cm["top10_overlap"], 0.10),
    ]:
        ax.bar(["Set 0", "Set 1", "Mean"], [val0, val1, vmean], color=COL_AA, alpha=0.85)
        if metric.startswith("Rank"):
            ax.axhline(rnd, color="#7f7f7f", linestyle="--", label=f"Random NDCG≈{rnd:.3f}")
        else:
            ax.axhline(rnd, color="#7f7f7f", linestyle="--", label="Random top-10≈0.10")
        ax.set_title(metric)
        ax.set_ylim(0, max(val0, val1, vmean, rnd) * 1.25)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 4 — Canonical A-full vs B-full baseline")
    fig.tight_layout()
    fig.savefig(OUT / "figure4_canonical_baseline.png", bbox_inches="tight")
    plt.close(fig)

    cross = json.loads((CANON / "analysis/merged_analysis.json").read_text(encoding="utf-8"))
    by_tau = cross["comparison_A_full_ref_vs_A_to_B_full_by_tau"]["by_tau"]
    taus_c = [r["tau"] for r in by_tau]
    nd_c = [r["rank_ndcg10_mean"] for r in by_tau]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(taus_c, nd_c, "o-", color=COL_AB, label="A-full vs A→B Full (NDCG@10)", ms=6)
    ax.axhline(cm["rank_ndcg10"], color=COL_AA, linestyle="--", label=f"A-full vs B-full ({cm['rank_ndcg10']:.3f})")
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Rank NDCG@10 (same A-full reference)")
    ax.invert_xaxis()
    ax.set_title("Figure 5 — Same-reference comparison across τ")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "figure5_same_ref_cross_tau.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"out": str(OUT), "figures": 6}, indent=2))


if __name__ == "__main__":
    main()