#!/usr/bin/env python3
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "raw_test_profiles.csv").open()))
assert len(rows) == 100000
means = {
    key: statistics.fmean(float(row[key]) for row in rows)
    for key in [
        "baseline_revenue",
        "caama_revenue",
        "caama_ir_regret",
        "caama_ex_post_ir_revenue",
        "welfare",
        "shuffled_pcor_ir_regret",
    ]
}
by_seed = defaultdict(list)
for row in rows:
    by_seed[int(row["seed"])].append(row)
checks = {
    "row_count": len(rows) == 100000,
    "five_seeds": sorted(by_seed) == [1, 7, 19, 41, 73],
    "twenty_thousand_profiles_per_seed": all(
        len(seed_rows) == 20000 for seed_rows in by_seed.values()
    ),
    "samplewise_revenue_bound": all(
        float(row["caama_revenue"])
        <= float(row["welfare"]) + float(row["caama_ir_regret"]) + 1e-6
        for row in rows
    ),
    "negative_control_increases_regret": (
        means["shuffled_pcor_ir_regret"] > means["caama_ir_regret"]
    ),
}
print(json.dumps({"means": means, "checks": checks}, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
