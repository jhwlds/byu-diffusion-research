#!/usr/bin/env python3
"""Build report figures from Phase 4 (dual projection) JSON — read-only."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP = Path("/yunity/jhwlds/experiments/fmnist_split_cross_injection")
P4 = EXP / "full_ckpt46000_phase4_proj2_20261110"
P3C = EXP / "full_ckpt46000_phase3_corrected_20261110"
OUT = Path("/yunity/jhwlds/report_github/byu-diffusion-research/visualizations/2026-11-10_cross-injection-gradient-proximity")
TAUS = [800, 600, 400, 200]
CASES = [
    ("A→A One", "A_to_A_one"),
    ("A→B One", "A_to_B_one"),
    ("A→A Full", "A_to_A_full"),
    ("A→B Full", "A_to_B_full"),
]
COL0 = "#1f77b4"
COL1 = "#ff7f0e"
COL_RB = "#7f7f7f"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def mean_metric(tau: int, case: str, grade: str, key: str, set_id: int) -> float:
    rep = load(P4 / f"tau_{tau}/projection_set_{set_id}/tau_report.json")
    return float(rep["cases"][case]["neighborhood"][grade][key]["mean"])


def overlap10(tau: int, case: str, set_id: int) -> float:
    rep = load(P4 / f"tau_{tau}/projection_set_{set_id}/tau_report.json")
    return float(rep["cases"][case]["neighborhood"]["top_k_overlap"]["top10"]["mean"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TAUS))
    labels = [str(t) for t in TAUS]
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 200})

    def series(case, grade, key, set_id):
        return [mean_metric(t, case, grade, key, set_id) for t in TAUS]

    # Fig1 — Rank NDCG@10 One-Step
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for case_l, case in CASES[:2]:
        ax.plot(x, series(case, "rank_grade", "ndcg@10", 0), "o-", label=f"{case_l} (Set 0)", markersize=6)
        ax.plot(x, series(case, "rank_grade", "ndcg@10", 1), "s--", label=f"{case_l} (Set 1)", markersize=5)
    rb = mean_metric(800, "A_to_A_one", "rank_grade", "random_baseline@10", 0)
    ax.axhline(rb, color=COL_RB, linestyle=":", label="Random baseline (~0.084)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Mean rank NDCG@10")
    ax.set_ylim(0, 1.02)
    ax.set_title("One-Step rank NDCG@10 (corrected sampling)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_one_step_ndcg10.png")
    plt.close(fig)

    # Fig2 — Cosine NDCG@10 One-Step (both projection sets)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for case_l, case in CASES[:2]:
        m0 = series(case, "cosine_grade", "ndcg@10", 0)
        m1 = series(case, "cosine_grade", "ndcg@10", 1)
        mu = [(a + b) / 2 for a, b in zip(m0, m1)]
        ax.plot(x, mu, "o-", label=case_l, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Mean cosine NDCG@10")
    ax.set_ylim(0.978, 1.001)
    ax.set_title("One-Step cosine NDCG@10 (set mean)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cosine_ndcg10_one_step.png")
    # Legacy filename kept for older links (same image)
    fig.savefig(OUT / "fig2_switching_gap.png")
    plt.close(fig)

    # Fig3 — Full rank NDCG@10
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for case_l, case in CASES[2:]:
        ax.plot(x, series(case, "rank_grade", "ndcg@10", 0), "o-", label=f"{case_l} Set 0", markersize=6)
        ax.plot(x, series(case, "rank_grade", "ndcg@10", 1), "s--", label=f"{case_l} Set 1", markersize=5)
    ax.axhline(rb, color=COL_RB, linestyle=":", label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel("Mean rank NDCG@10")
    ax.set_ylim(0, 0.65)
    ax.set_title("Full-denoising rank NDCG@10 (t=0 probe)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_full_ndcg10.png")
    plt.close(fig)

    # Fig4 — Rank NDCG@10 vs Top-10 overlap (One-Step, set means)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)

    def mean_sets(fn):
        return [(fn(t, 0) + fn(t, 1)) / 2 for t in TAUS]

    for case_l, case in CASES[:2]:
        ax1.plot(
            x,
            mean_sets(lambda t, s: mean_metric(t, case, "rank_grade", "ndcg@10", s)),
            "o-",
            label=case_l,
            markersize=6,
        )
    for case_l, case in CASES[:2]:
        ax2.plot(x, mean_sets(lambda t, s: overlap10(t, case, s)), "s-", label=case_l, markersize=6)
    ax1.set_ylabel("Rank NDCG@10")
    ax1.set_ylim(0.6, 1.02)
    ax1.set_title("One-Step NDCG")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax2.set_ylabel("Top-10 overlap")
    ax2.set_ylim(0.55, 1.02)
    ax2.set_title("One-Step overlap")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("τ")
    fig.suptitle("One-Step rank NDCG vs neighborhood overlap", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ndcg_vs_overlap.png", bbox_inches="tight")
    plt.close(fig)

    # Fig5 — One-Step vs Full rank NDCG@10 (A→A and A→B)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)
    for ax, route, one_c, full_c in (
        (axes[0], "A→A", "A_to_A_one", "A_to_A_full"),
        (axes[1], "A→B", "A_to_B_one", "A_to_B_full"),
    ):
        one = mean_sets(lambda t, s: mean_metric(t, one_c, "rank_grade", "ndcg@10", s))
        full = mean_sets(lambda t, s: mean_metric(t, full_c, "rank_grade", "ndcg@10", s))
        ax.plot(x, one, "o-", label="One-Step", markersize=6)
        ax.plot(x, full, "s-", label="Full (t=0)", markersize=6)
        ax.set_title(route)
        ax.set_ylabel("Rank NDCG@10")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("τ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("One-Step vs full-denoising rank NDCG@10", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_one_vs_full_ndcg10.png")
    plt.close(fig)

    # Supplementary — projection stability |rank@10 set0 − set1|
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.18
    for j, (case_l, case) in enumerate(CASES):
        diffs = [
            abs(mean_metric(t, case, "rank_grade", "ndcg@10", 0) - mean_metric(t, case, "rank_grade", "ndcg@10", 1))
            for t in TAUS
        ]
        ax.bar(x + (j - 1.5) * width, diffs, width=width, label=case_l)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injection τ")
    ax.set_ylabel(r"$|\mathrm{NDCG}_{10}^{(0)} - \mathrm{NDCG}_{10}^{(1)}|$ (rank)")
    ax.set_title("Projection sensitivity (two seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_projection_stability.png")
    plt.close(fig)

    # Flow diagram (unchanged schematic)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, "ε\n(fixed pool)"),
        (1.8, 1.2, "Model A\nDDIM → x_τ^A"),
        (3.6, 1.2, "Ref gradients\n@ τ (A)"),
        (5.4, 2.0, "A→A / A→B\nOne DDIM step"),
        (5.4, 0.5, "A→A / A→B\nFull denoise"),
        (7.6, 2.0, "Gradients\n@ τ′ (one)"),
        (7.6, 0.5, "Gradients\n@ t=0 (full)"),
        (9.0, 1.2, "NDCG /\noverlap"),
    ]
    for x0, y0, text in boxes:
        w, h = 1.35, 0.85
        rect = plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9)
    fig.suptitle("Cross-injection experiment flow (shared x_τ^A)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig0_experiment_flow.png")
    plt.close(fig)

    p400 = P3C / "tau_400"
    if p400.is_symlink():
        p400 = p400.resolve()
    grid = p400 / "plots" / "grid_AA_AB_pairs.png"
    if grid.is_file():
        shutil.copy2(grid, OUT / "grid_AA_AB_pairs_tau400.png")
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
