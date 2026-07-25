#!/usr/bin/env python3
"""Generate report figures from cumulative machine-readable evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
IMAGES = REPORT / "images"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"

COLORS = {
    "paper": "#687386",
    "baseline": "#5975A4",
    "caama": "#D65F5F",
    "support": "#55A868",
    "blocked": "#C08A24",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.edgecolor": "#c7ced8",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9,
        "grid.color": "#e1e6ed",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def _save(fig: plt.Figure, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        IMAGES / name,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "OpenResearch cumulative evidence"},
    )
    plt.close(fig)


def headline_claim4() -> None:
    summary = json.loads(
        (
            ARTIFACTS
            / "claim_4"
            / "route_6_exact_amenunet_five_seed"
            / "aggregate_summary.json"
        ).read_text()
    )
    labels = ["Randomized AMA", "CA-AMA"]
    paper = [3.1363, 3.6205]
    observed = [
        summary["baseline_revenue"]["mean"],
        summary["caama_revenue"]["mean"],
    ]
    errors = [
        0.0,
        summary["caama_revenue"]["ci95_high"] - observed[1],
    ]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(
        x - width / 2,
        paper,
        width,
        label="Paper Table 1",
        color=COLORS["paper"],
    )
    ax.bar(
        x + width / 2,
        observed,
        width,
        yerr=errors,
        capsize=4,
        label="Observed, five seeds",
        color=[COLORS["baseline"], COLORS["caama"]],
    )
    for position, value in zip(x - width / 2, paper):
        ax.text(position, value + 0.035, f"{value:.4f}", ha="center")
    for position, value in zip(x + width / 2, observed):
        ax.text(position, value + 0.035, f"{value:.4f}", ha="center")
    gain = observed[1] - observed[0]
    ax.text(
        0.5,
        0.15,
        f"Observed CA gain: +{gain:.4f}\n95% CI: [0.4797, 0.4955]",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#263445",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#c7ced8"},
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Expected revenue")
    ax.set_ylim(0, 4.25)
    ax.set_title("Claim 4: full 3-bidder × 10-item Dirichlet setting")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    _save(fig, "claim4_headline.svg")


def claim4_seed_stability() -> None:
    summary = json.loads(
        (
        ARTIFACTS
        / "claim_4"
        / "route_6_exact_amenunet_five_seed"
        / "per_seed_summary.json"
        ).read_text()
    )
    seeds = [int(seed) for seed in sorted(summary, key=int)]
    revenues = [
        summary[str(seed)]["caama_revenue"]["mean"] for seed in seeds
    ]
    regrets = [
        summary[str(seed)]["caama_ir_regret"]["mean"] for seed in seeds
    ]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.6))
    left.plot(
        seeds,
        revenues,
        "o-",
        color=COLORS["caama"],
        linewidth=1.8,
        label="Observed CA-AMA",
    )
    left.axhline(3.6205, color=COLORS["paper"], linestyle="--", label="Paper")
    left.fill_between(
        [min(seeds), max(seeds)],
        [3.6205 * 0.95] * 2,
        [3.6205 * 1.05] * 2,
        color=COLORS["paper"],
        alpha=0.12,
        label="±5% band",
    )
    left.set_title("Revenue stability")
    left.set_xlabel("Training seed")
    left.set_ylabel("CA-AMA revenue")
    left.grid()
    left.legend(fontsize=8)

    right.plot(
        seeds,
        regrets,
        "o-",
        color=COLORS["support"],
        linewidth=1.8,
        label="Observed IR regret",
    )
    right.axhline(0.0031, color=COLORS["paper"], linestyle="--", label="Paper")
    right.set_title("Held-out IR control")
    right.set_xlabel("Training seed")
    right.set_ylabel("Mean ex-post IR regret")
    right.set_ylim(0.0, 0.0072)
    right.grid()
    right.legend(fontsize=8)
    fig.suptitle(
        "Five exact released-AMenuNet training seeds",
        fontweight="bold",
    )
    _save(fig, "claim4_seed_stability.svg")


def theory_scope() -> None:
    raw = json.loads(
        (ARTIFACTS / "claim_1" / "raw_results.json").read_text()
    )
    rows = raw["rows"]
    by_n: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        by_n.setdefault(int(row["n_bidders"]), []).append(
            (
                float(row["target_delta"]),
                float(row["upper_bound_ratio"]),
            )
        )
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for n, points in by_n.items():
        points.sort(reverse=True)
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            "o-",
            alpha=0.75,
            label=f"n={n}",
        )
    deltas = sorted({float(row["target_delta"]) for row in rows})
    ax.plot(deltas, deltas, "k--", label="required upper bound δ")
    ax.axhline(
        1.0,
        color=COLORS["caama"],
        linewidth=2,
        label="n=1: D-AMA / REV = 1",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Requested factor δ")
    ax.set_ylabel("Certified D-AMA upper-bound ratio")
    ax.set_title("The n≥2 construction works; the literal n=1 scope does not")
    ax.grid(which="both")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, "theory_scope.svg")


def claim5_context() -> None:
    bound = json.loads(
        (
            ARTIFACTS
            / "claim_5"
            / "route_4_falsification_audit"
            / "raw_bound_results.json"
        ).read_text()
    )
    welfare = bound["distribution"]["expected_total_welfare_upper_bound"][
        "decimal"
    ]
    labels = ["Randomized AMA", "CA-AMA"]
    paper = [1.7135, 1.9359]
    pilot = [1.480823, 1.512781]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(
        x - width / 2,
        paper,
        width,
        color=COLORS["paper"],
        label="Paper Table 1",
    )
    ax.bar(
        x + width / 2,
        pilot,
        width,
        color=[COLORS["baseline"], COLORS["caama"]],
        label="CPU-upgrade pilot",
    )
    ax.axhline(
        welfare,
        color=COLORS["support"],
        linestyle="--",
        linewidth=1.8,
        label=f"Exact welfare bound {welfare:.4f}",
    )
    for position, value in zip(x - width / 2, paper):
        ax.text(position, value + 0.035, f"{value:.4f}", ha="center")
    for position, value in zip(x + width / 2, pilot):
        ax.text(position, value + 0.035, f"{value:.4f}", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Expected revenue")
    ax.set_ylim(0, 2.85)
    ax.set_title("Claim 5: pilot underfits; exact bounds do not falsify the paper")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    _save(fig, "claim5_context.svg")


def main() -> None:
    _style()
    headline_claim4()
    claim4_seed_stability()
    theory_scope()
    claim5_context()
    print(f"WROTE_REPORT_FIGURES={len(list(IMAGES.glob('*.svg')))}")


if __name__ == "__main__":
    main()
