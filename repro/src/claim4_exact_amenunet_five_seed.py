#!/usr/bin/env python3
"""Five-seed, paper-scale verification route for Table-1 Claim 4."""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psutil
import torch

import empirical_train as et
from claim4_exact_amenunet import (
    FIXED_COMMAND,
    PAPER_SHA256,
    PAPER_URL,
    UPSTREAM_SHA,
    _git_sha,
    _sha256,
    _write_csv,
    _write_json,
    _write_text,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "repro" / "config" / "empirical_train.json"
ROUTE = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "route_6_exact_amenunet_five_seed"
)
TARGETS = {
    "baseline_revenue": 3.1363,
    "caama_revenue": 3.6205,
    "caama_ir_regret": 0.0031,
    "caama_ex_post_ir_revenue": 3.5623,
}


def _worker(
    config: dict[str, Any], seed: int, worker_dir_text: str
) -> dict[str, Any]:
    """Run one deterministic seed and persist raw evidence before returning."""
    worker_dir = Path(worker_dir_text)
    started = time.perf_counter()
    try:
        torch.set_num_threads(int(config["torch_threads_per_worker"]))
        torch.set_num_interop_threads(1)
        print(
            f"CLAIM_4_WORKER_START seed={seed} "
            f"threads={torch.get_num_threads()}",
            flush=True,
        )
        raw, summary, controls, curves = et.run_seed(config, seed)
        _write_csv(worker_dir / "raw_test_samples.csv", raw)
        _write_csv(worker_dir / "learning_curves.csv", curves)
        _write_json(worker_dir / "summary.json", summary)
        _write_json(worker_dir / "negative_control_output.json", controls)
        elapsed = time.perf_counter() - started
        _write_json(
            worker_dir / "runtime.json",
            {
                "seed": seed,
                "elapsed_seconds": elapsed,
                "torch_threads": torch.get_num_threads(),
            },
        )
        print(
            "CLAIM_4_WORKER_DONE "
            f"seed={seed} baseline={summary['baseline_revenue']['mean']:.6f} "
            f"caama={summary['caama_revenue']['mean']:.6f} "
            f"regret={summary['caama_ir_regret']['mean']:.6f} "
            f"ex_post={summary['caama_ex_post_ir_revenue']['mean']:.6f} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )
        return {
            "seed": seed,
            "summary": summary,
            "controls": controls,
            "elapsed_seconds": elapsed,
            "raw_rows": len(raw),
        }
    except BaseException as error:
        _write_json(
            worker_dir / "worker_error.json",
            {
                "seed": seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
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
"""


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
independent = json.loads((here / "independent_checker_output.json").read_text())
negative = json.loads((here / "negative_control_verifier_output.json").read_text())
checks = dict(independent["checks"])
checks["independent_checker_passed"] = (
    independent["all_checks_pass"]
    and independent["raw_rows"] == 100000
)
checks["negative_control_verifier_passed"] = (
    negative["returncode"] == 0
    and negative["all_checks_pass"]
)
ok = all(checks.values())
verdict = "VERIFIED" if ok else "BLOCKED"
print(json.dumps({
    "claim": 4,
    "verdict": verdict,
    "ok": ok,
    "checks": checks,
}, sort_keys=True))
sys.exit(0 if ok else 1)
"""


def _negative_control_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
independent = json.loads((here / "independent_checker_output.json").read_text())
checks = {
    "zero_pcor_ablation_removes_revenue_ci": independent["checks"][
        "zero_pcor_ablation_removes_revenue_ci"
    ],
    "rival_reversal_increases_regret_ci": independent["checks"][
        "rival_reversal_increases_regret_ci"
    ],
}
result = {"checks": checks, "all_checks_pass": all(checks.values())}
print(json.dumps(result, sort_keys=True))
sys.exit(0 if result["all_checks_pass"] else 1)
"""


def _write_documentation(
    config: dict[str, Any],
    elapsed: float,
    worker_results: list[dict[str, Any]],
) -> None:
    _write_json(
        ROUTE / "claim_contract.json",
        {
            "claim": 4,
            "exact_statement": (
                "For Dirichlet Value Share(alpha=0.5), 3 bidders x 10 "
                "items, Table 1 reports randomized AMA revenue 3.1363, "
                "CA-AMA revenue 3.6205, IR regret 0.0031, and ex-post-IR "
                "revenue 3.5623, averaged over five seeds."
            ),
            "paper_result": TARGETS,
            "paper_source": PAPER_URL,
            "paper_sha256": PAPER_SHA256,
            "anchors": ["S4.T1", "S5.SS1", "S5.p3", "A1.SS4"],
            "quantifiers": {
                "training_seeds": [1, 2, 3, 4, 5],
                "updates_per_mechanism": 32_000,
                "test_profiles_per_seed": 20_000,
            },
            "pre_registered_tolerances": config["verification_tolerances"],
            "verdict_rule": (
                "VERIFIED only if every independent numeric, uncertainty, "
                "integrity, and negative-control check passes; otherwise "
                "BLOCKED. Optimization mismatch is not falsification."
            ),
            "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
        },
    )
    _write_text(
        ROUTE / "source_audit.md",
        f"""# Claim 4 source audit

- Paper: `{PAPER_URL}`
- Retrieved: `2026-07-23T15:56:49Z`
- Paper SHA-256: `{PAPER_SHA256}`
- Released-code commit: `{UPSTREAM_SHA}`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1`
- Implementation anchors: `S5.p3`, `A1.SS4`

The source states `T_j ~ U[0.5,1]`, symmetric
`Dirichlet(alpha,...,alpha)` shares, and `v_ij = w_ij T_j`. Table 1
reports randomized AMA `3.1363`, CA-AMA `3.6205`, IR regret `0.0031`,
and ex-post-IR revenue `3.5623` for alpha `0.5`, `3 x 10`, averaged
over five seeds.

Section 5 specifies 32,000 total iterations per mechanism, batch 1,024
for larger settings, mutual/post balance, softmax temperature 500, and
a fixed 20,000-profile test set. Algorithm 1 requires exact argmax AMA
payments and utilities in post-training.

Table 3 and the released shell matrix omit `3 x 10` and instead list a
transposed `10 x 3` setting. Menu size 2,048 and initial gamma 8 are
therefore explicit inferences, not paper-stated `3 x 10` facts.
""",
    )
    _write_text(
        ROUTE / "method.md",
        f"""# Claim 4 method

- Five deterministic training seeds: `1,2,3,4,5`.
- Literal Dirichlet Value Share(alpha=0.5), 3 bidders x 10 items.
- Released dropout-free AMenuNet transformer parameterization.
- Menu size 2,048; initial gamma 8; gamma update 0.01; cap 20.
- Randomized AMA: 32,000 optimizer updates.
- CA-AMA: 16,000 mutual + 16,000 pCor-only post updates.
- Batch 1,024; softmax temperature 500; allocation temperature 10.
- Fixed generator-seed-2002 test set with 20,000 profiles per seed.
- Paper-stated three-linear-layer, rival-only ReLU pCor network.
- Exact hard-argmax AMA outcomes/payments in post-training and evaluation.
- {config['parallel_workers']} isolated spawn workers, each restricted to
  {config['torch_threads_per_worker']} PyTorch CPU threads.
- Uncertainty is a two-sided 95% Student-t interval over five seed means.
- Independent verification recomputes all metrics from 100,000 raw rows.
""",
    )
    _write_text(
        ROUTE / "limitations_and_deviations.md",
        """# Limitations and deviations

- Menu size 2,048 and initial gamma 8 are inferred from the conflicting
  transposed `10 x 3` row; no released `3 x 10` checkpoint or command exists.
- "32,000 iterations" is interpreted as optimizer updates. Released scripts
  use outer data-generation loops containing multiple minibatch updates, so
  the paper and public code do not expose one unambiguous counter.
- The paper states a three-layer ReLU pCor network. Released mutual training
  instead uses max-minus-max, while released post-training uses ReLU.
- Constant-context vectorization is algebraically regression-tested against
  released AMenuNet but is a CPU execution optimization.
- Concurrent workers affect wall-clock time, not seeds, samples, or contracts.
- A divergent non-convex optimization outcome is BLOCKED, not FALSIFIED.
""",
    )
    _write_json(
        ROUTE / "exact_command_environment.json",
        {
            "fixed_command": FIXED_COMMAND,
            "git_sha": _git_sha(),
            "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "memory_bytes": psutil.virtual_memory().total,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "seeds": config["seeds"],
            "parallel_workers": config["parallel_workers"],
            "torch_threads_per_worker": config["torch_threads_per_worker"],
            "elapsed_seconds": elapsed,
            "worker_results": worker_results,
        },
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    if seeds != [1, 2, 3, 4, 5]:
        raise ValueError(f"five-seed route requires seeds 1..5, got {seeds}")
    if int(config["parallel_workers"]) != 5:
        raise ValueError("five-seed route requires five isolated workers")

    print(
        "CLAIM_4_CUMULATIVE_REGRESSION="
        "accepted_theory_rerun_blocked_empirical_preserved",
        flush=True,
    )
    print(
        "CLAIM_4_PRIOR_BLOCKED_RUNS="
        "8770c5f1-7f57-4383-8caf-c69eb475714c,"
        "233ef5c5-063e-40d8-a810-ef661f153826,"
        "298cd9b9-0ac9-472b-b929-d56a9ac3613b",
        flush=True,
    )
    _write_json(ROUTE / "config.json", config)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(config["parallel_workers"]),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _worker, config, seed, str(ROUTE / f"seed_{seed}")
            ): seed
            for seed in seeds
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                results.append(future.result())
            except BaseException as error:
                raise RuntimeError(f"Claim 4 worker seed {seed} failed") from error
    results.sort(key=lambda result: int(result["seed"]))
    if any(result["raw_rows"] != 20_000 for result in results):
        raise RuntimeError("at least one seed did not write 20,000 raw rows")

    summaries = {
        str(result["seed"]): result["summary"] for result in results
    }
    controls = {
        str(result["seed"]): result["controls"] for result in results
    }
    aggregate = et.aggregate_seed_results(summaries)
    elapsed = time.perf_counter() - started
    _write_json(ROUTE / "per_seed_summary.json", summaries)
    _write_json(ROUTE / "aggregate_summary.json", aggregate)
    _write_json(ROUTE / "negative_control_output.json", controls)
    _write_documentation(config, elapsed, results)

    _write_text(
        ROUTE / "independent_checker.py", _independent_checker_source()
    )
    independent = subprocess.run(
        [sys.executable, str(ROUTE / "independent_checker.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    independent_payload = (
        json.loads(independent.stdout)
        if independent.stdout.strip()
        else {"all_checks_pass": False, "checks": {}}
    )
    independent_payload["returncode"] = independent.returncode
    independent_payload["stderr"] = independent.stderr.strip()
    _write_json(
        ROUTE / "independent_checker_output.json", independent_payload
    )
    _write_json(
        ROUTE / "verification_criteria.json",
        {
            "paper_targets": TARGETS,
            "pre_registered_tolerances": config["verification_tolerances"],
            "independent_checks": independent_payload.get("checks", {}),
            "all_verification_checks_pass": independent_payload.get(
                "all_checks_pass", False
            ),
        },
    )

    _write_text(
        ROUTE / "negative_control_verifier.py",
        _negative_control_verifier_source(),
    )
    negative = subprocess.run(
        [sys.executable, str(ROUTE / "negative_control_verifier.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    negative_payload = (
        json.loads(negative.stdout)
        if negative.stdout.strip()
        else {"all_checks_pass": False, "checks": {}}
    )
    negative_payload["returncode"] = negative.returncode
    negative_payload["stderr"] = negative.stderr.strip()
    _write_json(
        ROUTE / "negative_control_verifier_output.json",
        negative_payload,
    )

    _write_text(ROUTE / "claim_verifier.py", _claim_verifier_source())
    verifier = subprocess.run(
        [sys.executable, str(ROUTE / "claim_verifier.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    verifier_payload = (
        json.loads(verifier.stdout)
        if verifier.stdout.strip()
        else {"verdict": "BLOCKED", "ok": False, "checks": {}}
    )
    verifier_payload["returncode"] = verifier.returncode
    verifier_payload["stderr"] = verifier.stderr.strip()
    _write_json(
        ROUTE / "claim_verifier_output.json", verifier_payload
    )
    verdict = (
        "VERIFIED"
        if verifier.returncode == 0
        and verifier_payload.get("verdict") == "VERIFIED"
        else "BLOCKED"
    )
    aggregate_means = independent_payload.get("aggregate_means", {})
    _write_text(
        ROUTE / "EVAL.md",
        f"""# Claim 4 evaluation

- Verdict: **{verdict}**
- Randomized AMA: `{aggregate_means.get('baseline_revenue', float('nan')):.6f}`
  (paper `3.1363`)
- CA-AMA: `{aggregate_means.get('caama_revenue', float('nan')):.6f}`
  (paper `3.6205`)
- IR regret: `{aggregate_means.get('caama_ir_regret', float('nan')):.6f}`
  (paper `0.0031`)
- Ex-post-IR revenue:
  `{aggregate_means.get('caama_ex_post_ir_revenue', float('nan')):.6f}`
  (paper `3.5623`)
- Claim verifier exit: `{verifier.returncode}`
- Independent checker exit: `{independent.returncode}`
- Negative-control verifier exit: `{negative.returncode}`
""",
    )
    print(
        "CLAIM_4_EXACT_FIVE_SEED "
        f"verdict={verdict} "
        f"baseline={aggregate_means.get('baseline_revenue', float('nan')):.6f} "
        f"caama={aggregate_means.get('caama_revenue', float('nan')):.6f} "
        f"regret={aggregate_means.get('caama_ir_regret', float('nan')):.6f} "
        "ex_post="
        f"{aggregate_means.get('caama_ex_post_ir_revenue', float('nan')):.6f} "
        f"claim_verifier_exit={verifier.returncode} "
        f"negative_control_exit={negative.returncode} "
        f"seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
