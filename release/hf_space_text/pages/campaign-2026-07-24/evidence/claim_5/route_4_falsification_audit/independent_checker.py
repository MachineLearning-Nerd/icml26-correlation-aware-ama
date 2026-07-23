#!/usr/bin/env python3
import json
from fractions import Fraction
from pathlib import Path

here = Path(__file__).resolve().parent
raw = json.loads((here / "raw_bound_results.json").read_text())
alpha = Fraction(3, 5)
correlated = Fraction(21, 40)
independent = Fraction(1, 2) + Fraction(1, 4) ** 2 / 6
expected_welfare = 5 * (
    alpha * correlated + (1 - alpha) * independent
)
reported = {
    "baseline": Fraction(17135, 10000),
    "caama": Fraction(19359, 10000),
    "regret": Fraction(52, 10000),
    "ex_post": Fraction(18553, 10000),
}
checks = {
    "expected_welfare_exact": (
        raw["distribution"]["expected_total_welfare_upper_bound"]["exact"]
        == f"{expected_welfare.numerator}/{expected_welfare.denominator}"
    ),
    "baseline_feasible": reported["baseline"] <= expected_welfare,
    "caama_feasible_with_regret": (
        reported["caama"] <= expected_welfare + reported["regret"]
    ),
    "ex_post_feasible": reported["ex_post"] <= expected_welfare,
}
result = {
    "expected_welfare_exact": (
        f"{expected_welfare.numerator}/{expected_welfare.denominator}"
    ),
    "checks": checks,
    "valid_falsification_found": not all(checks.values()),
}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
