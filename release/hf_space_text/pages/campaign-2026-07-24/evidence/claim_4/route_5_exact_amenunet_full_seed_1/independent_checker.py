#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "raw_test_samples.csv").open()))
summary = json.loads((here / "summary.json").read_text())
config = json.loads((here / "config.json").read_text())
keys = (
    "baseline_revenue",
    "caama_revenue",
    "caama_ir_regret",
    "caama_ex_post_ir_revenue",
)
means = {
    key: sum(float(row[key]) for row in rows) / len(rows)
    for key in keys
}
checks = {
    "raw_rows": len(rows) == 20000,
    "means_match": all(
        math.isclose(
            means[key], float(summary[key]["mean"]), rel_tol=0, abs_tol=1e-10
        )
        for key in keys
    ),
    "paper_scale_updates": (
        config["baseline_updates"] == 32000
        and config["mutual_updates"] == 16000
        and config["post_updates"] == 16000
    ),
    "literal_setting": (
        config["distribution"] == "dirichlet_value_share"
        and config["n_bidders"] == 3
        and config["n_items"] == 10
        and config["alpha"] == 0.5
    ),
}
print(json.dumps({
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "recomputed_means": means,
}, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
