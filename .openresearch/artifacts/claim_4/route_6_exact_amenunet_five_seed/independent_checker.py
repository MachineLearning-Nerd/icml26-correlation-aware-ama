#!/usr/bin/env python3
import csv
import json
import math
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
config = json.loads((here / "config.json").read_text())
targets = {
    "baseline_revenue": 3.1363,
    "caama_revenue": 3.6205,
    "caama_ir_regret": 0.0031,
    "caama_ex_post_ir_revenue": 3.5623,
}
keys = tuple(targets)
by_seed = {}
for seed in (1, 2, 3, 4, 5):
    path = here / f"seed_{seed}" / "raw_test_samples.csv"
    by_seed[seed] = list(csv.DictReader(path.open()))

seed_means = {
    key: [
        statistics.fmean(float(row[key]) for row in by_seed[seed])
        for seed in sorted(by_seed)
    ]
    for key in keys
}
zero_pcor = [
    statistics.fmean(float(row["zero_pcor_revenue"]) for row in by_seed[seed])
    for seed in sorted(by_seed)
]
shuffled_regret = [
    statistics.fmean(
        float(row["shuffled_pcor_ir_regret"]) for row in by_seed[seed]
    )
    for seed in sorted(by_seed)
]
t4 = 2.7764451051977987

def interval(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_error": se,
        "ci95_low": mean - t4 * se,
        "ci95_high": mean + t4 * se,
        "values": values,
    }

paired_improvement = [
    seed_means["caama_revenue"][index]
    - seed_means["baseline_revenue"][index]
    for index in range(5)
]
zero_pcor_effect = [
    seed_means["caama_revenue"][index] - zero_pcor[index]
    for index in range(5)
]
rival_reversal_effect = [
    shuffled_regret[index] - seed_means["caama_ir_regret"][index]
    for index in range(5)
]
aggregate_means = {
    key: statistics.fmean(values) for key, values in seed_means.items()
}
relative_errors = {
    key: abs(aggregate_means[key] - targets[key]) / targets[key]
    for key in (
        "baseline_revenue",
        "caama_revenue",
        "caama_ex_post_ir_revenue",
    )
}
tolerances = config["verification_tolerances"]
paired_interval = interval(paired_improvement)
zero_interval = interval(zero_pcor_effect)
reversal_interval = interval(rival_reversal_effect)
checks = {
    "five_exact_seeds": config["seeds"] == [1, 2, 3, 4, 5],
    "five_raw_files": sorted(by_seed) == [1, 2, 3, 4, 5],
    "twenty_thousand_rows_per_seed": all(
        len(rows) == 20000 for rows in by_seed.values()
    ),
    "literal_setting": (
        config["distribution"] == "dirichlet_value_share"
        and config["n_bidders"] == 3
        and config["n_items"] == 10
        and config["alpha"] == 0.5
    ),
    "released_amenunet_parameterization": (
        config["parameterization"] == "amenunet_constant_context"
    ),
    "paper_scale_updates": (
        config["baseline_updates"] == 32000
        and config["mutual_updates"] == 16000
        and config["post_updates"] == 16000
    ),
    "paper_batch_and_test_sizes": (
        config["train_batch_size"] == 1024
        and config["eval_samples"] == 20000
    ),
    "baseline_within_tolerance": (
        relative_errors["baseline_revenue"]
        <= tolerances["revenue_relative_error"]
    ),
    "caama_within_tolerance": (
        relative_errors["caama_revenue"]
        <= tolerances["revenue_relative_error"]
    ),
    "ex_post_ir_within_tolerance": (
        relative_errors["caama_ex_post_ir_revenue"]
        <= tolerances["revenue_relative_error"]
    ),
    "ir_regret_within_tolerance": (
        abs(
            aggregate_means["caama_ir_regret"]
            - targets["caama_ir_regret"]
        )
        <= tolerances["ir_regret_absolute_error"]
    ),
    "paired_improvement_ci_excludes_zero": (
        paired_interval["ci95_low"]
        > tolerances["paired_improvement_ci95_low_minimum"]
    ),
    "zero_pcor_ablation_removes_revenue_ci": zero_interval["ci95_low"] > 0,
    "rival_reversal_increases_regret_ci": reversal_interval["ci95_low"] > 0,
}
result = {
    "raw_rows": sum(len(rows) for rows in by_seed.values()),
    "seed_counts": {
        str(seed): len(rows) for seed, rows in sorted(by_seed.items())
    },
    "seed_means": seed_means,
    "aggregate_means": aggregate_means,
    "relative_errors": relative_errors,
    "ir_regret_absolute_error": abs(
        aggregate_means["caama_ir_regret"] - targets["caama_ir_regret"]
    ),
    "paired_caama_minus_baseline": paired_interval,
    "negative_controls": {
        "zero_pcor_revenue_effect": zero_interval,
        "rival_reversal_regret_effect": reversal_interval,
    },
    "checks": checks,
    "all_checks_pass": all(checks.values()),
}
print(json.dumps(result, sort_keys=True))
sys.exit(0 if result["all_checks_pass"] else 1)
