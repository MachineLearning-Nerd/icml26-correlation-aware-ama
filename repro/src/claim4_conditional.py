#!/usr/bin/env python3
"""Full-scale conditional-support mechanism route for Table 1 Claim 4."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import scipy.stats


ROOT = Path(__file__).resolve().parents[2]
ROUTE = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "route_2_conditional_support"
)
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2602.09455"
PAPER_SHA256 = (
    "2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f"
)
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)
SEEDS = (1, 2, 3, 4, 5)
N_BIDDERS = 3
N_ITEMS = 10
ALPHA = 0.5
TRAIN_ITEM_SAMPLES = 100_000
TEST_PROFILES = 20_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_dirichlet_profiles(
    seed: int, profiles: int, items: int = N_ITEMS
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    totals = rng.uniform(0.5, 1.0, size=(profiles, items, 1))
    gamma = rng.gamma(
        shape=ALPHA, scale=1.0, size=(profiles, items, N_BIDDERS)
    )
    shares = gamma / gamma.sum(axis=-1, keepdims=True)
    return totals * shares


def conditional_utility_floor(
    values: np.ndarray, reserve: float
) -> np.ndarray:
    """Exact rival-only utility infimum for a separable reserve VCG auction."""
    floors = np.zeros_like(values)
    for bidder in range(N_BIDDERS):
        rivals = np.delete(values, bidder, axis=-1)
        rival_sum = rivals.sum(axis=-1)
        rival_max = rivals.max(axis=-1)
        own_support_infimum = np.maximum(0.5 - rival_sum, 0.0)
        floors[..., bidder] = np.maximum(
            own_support_infimum - np.maximum(reserve, rival_max), 0.0
        )
    return floors


def auction_metrics(
    values: np.ndarray, reserve: float, shuffled_floors: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    order = np.sort(values, axis=-1)
    top = order[..., -1]
    second = order[..., -2]
    winners = values.argmax(axis=-1)
    sold = top >= reserve
    item_payment = np.where(sold, np.maximum(reserve, second), 0.0)
    base_by_bidder = np.zeros_like(values)
    for bidder in range(N_BIDDERS):
        base_by_bidder[..., bidder] = np.where(
            sold & (winners == bidder), item_payment, 0.0
        )
    allocation_value = np.zeros_like(values)
    for bidder in range(N_BIDDERS):
        allocation_value[..., bidder] = np.where(
            sold & (winners == bidder), values[..., bidder], 0.0
        )
    floor = conditional_utility_floor(values, reserve)
    utility = allocation_value - base_by_bidder - floor
    bidder_utility = utility.sum(axis=1)
    base_profile = base_by_bidder.sum(axis=(1, 2))
    pcor_profile = floor.sum(axis=(1, 2))
    ca_profile = base_profile + pcor_profile
    welfare_profile = allocation_value.sum(axis=(1, 2))
    regret_profile = np.maximum(-bidder_utility, 0.0).sum(axis=1)
    ex_post_profile = np.where(
        bidder_utility >= -1e-12,
        (base_by_bidder + floor).sum(axis=1),
        0.0,
    ).sum(axis=1)
    result = {
        "baseline_revenue": base_profile,
        "caama_revenue": ca_profile,
        "caama_ir_regret": regret_profile,
        "caama_ex_post_ir_revenue": ex_post_profile,
        "welfare": welfare_profile,
        "pcor_revenue": pcor_profile,
        "minimum_bidder_utility": bidder_utility.min(axis=1),
    }
    if shuffled_floors is not None:
        shuffled_utility = (
            allocation_value - base_by_bidder - shuffled_floors
        ).sum(axis=1)
        result["shuffled_pcor_ir_regret"] = np.maximum(
            -shuffled_utility, 0.0
        ).sum(axis=1)
    return result


def optimize_reserve(seed: int, for_caama: bool) -> tuple[float, float]:
    # Itemwise iid structure means a one-item training draw is sufficient.
    values = sample_dirichlet_profiles(
        seed=10_000 + seed, profiles=TRAIN_ITEM_SAMPLES, items=1
    )
    top = values[:, 0].max(axis=-1)
    second = np.sort(values[:, 0], axis=-1)[:, -2]
    floor_inputs = values if for_caama else None
    best_reserve = 0.0
    best_revenue = -math.inf
    for reserve in np.linspace(0.0, 1.0, 501):
        revenue = np.where(
            top >= reserve, np.maximum(reserve, second), 0.0
        )
        mean = float(revenue.mean())
        if floor_inputs is not None:
            mean += float(
                conditional_utility_floor(floor_inputs, float(reserve))
                .sum(axis=(1, 2))
                .mean()
            )
        if mean > best_revenue:
            best_revenue = mean
            best_reserve = float(reserve)
    return best_reserve, best_revenue


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    critical = float(scipy.stats.t.ppf(0.975, len(array) - 1))
    mean = float(array.mean())
    return {
        "count": len(array),
        "mean": mean,
        "std": float(array.std(ddof=1)),
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "seed_means": array.tolist(),
    }


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
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
"""


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
from pathlib import Path

here = Path(__file__).resolve().parent
criteria = json.loads((here / "verification_criteria.json").read_text())
independent = json.loads(
    (here / "independent_checker_output.json").read_text()
)
checks = dict(criteria["checks"])
checks["independent_checker"] = all(independent["checks"].values())
checks["method_matches_paper_neural_2048_menu"] = False
ok = all(checks.values())
print(json.dumps({
    "claim": 4,
    "verdict": "VERIFIED" if ok else "BLOCKED",
    "checks": checks,
}, sort_keys=True))
raise SystemExit(0 if ok else 1)
"""


def main() -> None:
    started = time.perf_counter()
    test_values = sample_dirichlet_profiles(
        seed=2_002, profiles=TEST_PROFILES
    )
    raw_rows: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}
    normal_regrets: list[float] = []
    shuffled_regrets: list[float] = []
    for seed in SEEDS:
        baseline_reserve, baseline_train_revenue = optimize_reserve(
            seed, for_caama=False
        )
        ca_reserve, ca_train_revenue = optimize_reserve(
            seed, for_caama=True
        )
        baseline_metrics = auction_metrics(test_values, baseline_reserve)
        floor = conditional_utility_floor(test_values, ca_reserve)
        shuffled_floor = floor[::-1].copy()
        ca_metrics = auction_metrics(
            test_values, ca_reserve, shuffled_floors=shuffled_floor
        )
        metrics = {
            "baseline_revenue": float(
                baseline_metrics["baseline_revenue"].mean()
            ),
            "caama_revenue": float(ca_metrics["caama_revenue"].mean()),
            "caama_ir_regret": float(
                ca_metrics["caama_ir_regret"].mean()
            ),
            "caama_ex_post_ir_revenue": float(
                ca_metrics["caama_ex_post_ir_revenue"].mean()
            ),
            "welfare": float(ca_metrics["welfare"].mean()),
            "pcor_revenue": float(ca_metrics["pcor_revenue"].mean()),
            "shuffled_pcor_ir_regret": float(
                ca_metrics["shuffled_pcor_ir_regret"].mean()
            ),
        }
        normal_regrets.append(metrics["caama_ir_regret"])
        shuffled_regrets.append(metrics["shuffled_pcor_ir_regret"])
        seed_summaries[str(seed)] = {
            "baseline_reserve": baseline_reserve,
            "caama_reserve": ca_reserve,
            "baseline_training_item_revenue": baseline_train_revenue,
            "caama_training_item_revenue": ca_train_revenue,
            **metrics,
        }
        for index in range(TEST_PROFILES):
            raw_rows.append(
                {
                    "seed": seed,
                    "profile_index": index,
                    "baseline_revenue": float(
                        baseline_metrics["baseline_revenue"][index]
                    ),
                    "caama_revenue": float(
                        ca_metrics["caama_revenue"][index]
                    ),
                    "caama_ir_regret": float(
                        ca_metrics["caama_ir_regret"][index]
                    ),
                    "caama_ex_post_ir_revenue": float(
                        ca_metrics["caama_ex_post_ir_revenue"][index]
                    ),
                    "welfare": float(ca_metrics["welfare"][index]),
                    "pcor_revenue": float(
                        ca_metrics["pcor_revenue"][index]
                    ),
                    "minimum_bidder_utility": float(
                        ca_metrics["minimum_bidder_utility"][index]
                    ),
                    "shuffled_pcor_ir_regret": float(
                        ca_metrics["shuffled_pcor_ir_regret"][index]
                    ),
                }
            )
        print(
            f"CLAIM_4_CONDITIONAL seed={seed} "
            f"baseline={metrics['baseline_revenue']:.6f} "
            f"caama={metrics['caama_revenue']:.6f} "
            f"regret={metrics['caama_ir_regret']:.6f} "
            f"reserves={baseline_reserve:.3f}/{ca_reserve:.3f}"
        )
    _write_csv(ROUTE / "raw_test_profiles.csv", raw_rows)
    aggregate = {
        metric: _summary(
            [seed_summaries[str(seed)][metric] for seed in SEEDS]
        )
        for metric in (
            "baseline_revenue",
            "caama_revenue",
            "caama_ir_regret",
            "caama_ex_post_ir_revenue",
            "welfare",
            "pcor_revenue",
        )
    }
    improvement = [
        seed_summaries[str(seed)]["caama_revenue"]
        - seed_summaries[str(seed)]["baseline_revenue"]
        for seed in SEEDS
    ]
    aggregate["paired_caama_minus_baseline"] = _summary(improvement)
    targets = {
        "baseline_revenue": 3.1363,
        "caama_revenue": 3.6205,
        "caama_ir_regret": 0.0031,
        "caama_ex_post_ir_revenue": 3.5623,
    }
    relative_errors = {
        key: abs(aggregate[key]["mean"] - value) / abs(value)
        for key, value in targets.items()
        if key != "caama_ir_regret"
    }
    regret_error = abs(
        aggregate["caama_ir_regret"]["mean"]
        - targets["caama_ir_regret"]
    )
    negative_effect = _summary(
        [
            shuffled_regrets[index] - normal_regrets[index]
            for index in range(len(SEEDS))
        ]
    )
    criteria = {
        "targets": targets,
        "relative_errors": relative_errors,
        "ir_regret_absolute_error": regret_error,
        "checks": {
            "five_training_seeds": len(SEEDS) == 5,
            "paper_scale_bidders_items": (
                N_BIDDERS == 3 and N_ITEMS == 10
            ),
            "fixed_test_size": TEST_PROFILES == 20_000,
            "baseline_within_5_percent": (
                relative_errors["baseline_revenue"] <= 0.05
            ),
            "caama_within_5_percent": (
                relative_errors["caama_revenue"] <= 0.05
            ),
            "ex_post_within_5_percent": (
                relative_errors["caama_ex_post_ir_revenue"] <= 0.05
            ),
            "ir_regret_within_0_003": regret_error <= 0.003,
            "paired_improvement_ci_excludes_zero": (
                aggregate["paired_caama_minus_baseline"]["ci95_low"] > 0
            ),
            "shuffled_rivals_increase_regret": (
                negative_effect["ci95_low"] > 0
            ),
        },
    }
    _write_json(ROUTE / "claim_contract.json", {
        "claim": 4,
        "paper_result": {
            "setting": "Dirichlet Value Share alpha=0.5, 3 bidders x 10 items",
            "randomized_ama_revenue": 3.1363,
            "caama_revenue": 3.6205,
            "caama_ir_regret": 0.0031,
            "caama_ex_post_ir_revenue": 3.5623,
        },
        "acceptance": (
            "Five-seed revenue means within 5%, regret within 0.003, "
            "positive paired CI, sensitive negative control, and the paper's "
            "neural 2048-menu method scope."
        ),
        "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
    })
    _write_text(ROUTE / "source_audit.md", f"""# Claim 4 source audit

- Source: `{PAPER_URL}`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `{PAPER_SHA256}`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1`
- Appendix hyperparameter anchor: Table 3

Tables 1 and 2 specify `3 x 10`. The released command matrix and Appendix
Table 3 instead contain `10 x 3` with menu 2048. The released `result.txt`
reproduces other Table 1 rows and confirms its convention is bidders first,
but it contains no `3_10` raw row. Revenue `3.6205` would also be impossible
with only three Dirichlet items because total welfare is at most 3. Therefore
this route uses three bidders and ten items; menu 2048 remains an inference,
not a fully specified released configuration.
""")
    _write_text(ROUTE / "method.md", """# Conditional-support method

- Full paper setting: three bidders, ten items, alpha=0.5.
- Five deterministic mechanism-training seeds and one fixed 20,000-profile
  evaluation set.
- A 501-point reserve grid is optimized from 100,000 iid item draws per seed.
- The baseline is a valid separable reserve VCG/AMA.
- CA-AMA adds the exact rival-only conditional utility infimum. Given two rival
  values with sum `s`, the own-value support infimum is `max(0.5-s, 0)`.
- This construction is DSIC and support-wise IR by definition; all inequalities
  are also checked samplewise.
- A reversed-profile pCor control destroys the rival matching and must increase
  IR regret.
""")
    _write_json(ROUTE / "per_seed_summary.json", seed_summaries)
    _write_json(ROUTE / "aggregate_summary.json", aggregate)
    _write_json(ROUTE / "verification_criteria.json", criteria)
    _write_json(ROUTE / "negative_control_output.json", {
        "normal_ir_regret_seed_means": normal_regrets,
        "shuffled_ir_regret_seed_means": shuffled_regrets,
        "paired_effect": negative_effect,
    })
    _write_text(
        ROUTE / "independent_checker.py", _independent_checker_source()
    )
    independent = subprocess.run(
        [sys.executable, str(ROUTE / "independent_checker.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if independent.returncode != 0:
        raise RuntimeError(f"independent checker failed: {independent.stderr}")
    _write_json(
        ROUTE / "independent_checker_output.json",
        json.loads(independent.stdout),
    )
    _write_text(ROUTE / "claim_verifier.py", _claim_verifier_source())
    verifier = subprocess.run(
        [sys.executable, str(ROUTE / "claim_verifier.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    _write_json(ROUTE / "verifier_output.json", {
        "returncode": verifier.returncode,
        "stdout": verifier.stdout.strip(),
        "stderr": verifier.stderr.strip(),
    })
    _write_json(ROUTE / "exact_command_environment.json", {
        "fixed_command": FIXED_COMMAND,
        "git_sha": _git_sha(),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "seeds": list(SEEDS),
        "test_seed": 2_002,
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_text(ROUTE / "limitations_and_deviations.md", """# Limitations and deviations

- This is a valid full-scale CA-AMA, but it is separable across items and uses
  an analytically derived payment rather than the paper's learned 2048-menu
  joint AMenuNet and three-layer pCor network.
- The route directly tests whether the reported revenue regime is attainable
  under the exact distribution; it does not reproduce the authors' missing
  3x10 checkpoints or their optimizer trajectory.
- The paper/release transpose discrepancy prevents exact recovery of the 3x10
  menu and gamma configuration from public artifacts.
""")
    verdict = "VERIFIED" if verifier.returncode == 0 else "BLOCKED"
    _write_text(ROUTE / "EVAL.md", f"""# Claim 4 conditional-support evaluation

- Verdict: **{verdict}**
- Baseline observed: `{aggregate['baseline_revenue']['mean']:.6f}` (paper `3.1363`)
- CA-AMA observed: `{aggregate['caama_revenue']['mean']:.6f}` (paper `3.6205`)
- IR regret observed: `{aggregate['caama_ir_regret']['mean']:.6f}` (paper `0.0031`)
- Ex-post-IR observed: `{aggregate['caama_ex_post_ir_revenue']['mean']:.6f}`
  (paper `3.5623`)
- Paired improvement CI:
  `[{aggregate['paired_caama_minus_baseline']['ci95_low']:.6f}, `
  `{aggregate['paired_caama_minus_baseline']['ci95_high']:.6f}]`
- Verifier exit: `{verifier.returncode}`

The method-scope check intentionally fails unless the released 3x10 neural
configuration is recoverable and run. Matching mechanism-level numbers alone
cannot erase that deviation.
""")
    print(
        "CLAIM_4_CONDITIONAL_AGGREGATE "
        f"baseline={aggregate['baseline_revenue']['mean']:.6f} "
        f"caama={aggregate['caama_revenue']['mean']:.6f} "
        f"regret={aggregate['caama_ir_regret']['mean']:.6f} "
        f"ex_post={aggregate['caama_ex_post_ir_revenue']['mean']:.6f}"
    )
    print(f"CLAIM_4_VERDICT={verdict}")
    print(f"CLAIM_4_VERIFIER_EXIT={verifier.returncode}")
    print(f"CLAIM_4_ROUTE_RUNTIME_SECONDS={time.perf_counter()-started:.6f}")


if __name__ == "__main__":
    main()
