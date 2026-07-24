#!/usr/bin/env python3
import json
from pathlib import Path

here = Path(__file__).resolve().parent
aggregate = json.loads((here / "aggregate_summary.json").read_text())
paper = {"baseline": 3.1363, "caama": 3.6205, "regret": 0.0031, "ex_post": 3.5623}
checks = {
    "exact_released_2048_menu_core_available": False,
    "five_independent_training_seeds": aggregate["seed_count"] == 5,
    "baseline_within_5pct": abs(
        aggregate["baseline_revenue"]["mean"] - paper["baseline"]
    ) / paper["baseline"] <= 0.05,
    "caama_within_5pct": abs(
        aggregate["caama_revenue"]["mean"] - paper["caama"]
    ) / paper["caama"] <= 0.05,
    "regret_at_paper_scale": (
        aggregate["caama_ir_regret"]["mean"] <= paper["regret"] + 0.001
    ),
    "ex_post_within_5pct": abs(
        aggregate["caama_ex_post_ir_revenue"]["mean"] - paper["ex_post"]
    ) / paper["ex_post"] <= 0.05,
    "positive_gain_ci95": (
        aggregate["caama_minus_baseline"]["ci95_low"] > 0
    ),
    "negative_control": aggregate["negative_control_pass"],
}
print(json.dumps({"paper": paper, "checks": checks}, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
