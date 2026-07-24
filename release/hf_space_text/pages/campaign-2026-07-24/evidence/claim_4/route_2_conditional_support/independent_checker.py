#!/usr/bin/env python3
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "raw_test_profiles.csv").open()))
by_seed = defaultdict(list)
for row in rows:
    by_seed[int(row["seed"])].append(row)
assert sorted(by_seed) == [1, 2, 3, 4, 5]
assert all(len(seed_rows) == 20000 for seed_rows in by_seed.values())
metrics = [
    "baseline_revenue",
    "caama_revenue",
    "caama_ir_regret",
    "caama_ex_post_ir_revenue",
]
means = {
    metric: [
        statistics.fmean(float(row[metric]) for row in by_seed[seed])
        for seed in sorted(by_seed)
    ]
    for metric in metrics
}
checks = {
    "row_count": len(rows) == 100000,
    "samplewise_revenue_below_welfare": all(
        float(row["caama_revenue"])
        <= float(row["welfare"]) + float(row["caama_ir_regret"]) + 1e-9
        for row in rows
    ),
    "samplewise_nonnegative_conditional_utility": all(
        float(row["minimum_bidder_utility"]) >= -1e-9 for row in rows
    ),
}
print(json.dumps({"seed_means": means, "checks": checks}, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
