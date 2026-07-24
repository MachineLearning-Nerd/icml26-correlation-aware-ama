#!/usr/bin/env python3
"""Paper-scale single-seed convergence gate for Table-1 Claim 4."""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
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
    / "route_5_exact_amenunet_full_seed_1"
)
TARGETS = {
    "baseline_revenue": 3.1363,
    "caama_revenue": 3.6205,
    "caama_ir_regret": 0.0031,
    "caama_ex_post_ir_revenue": 3.5623,
}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _independent_checks(
    config: dict[str, Any],
    raw: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    recomputed = {
        key: _mean(raw, key)
        for key in (
            "baseline_revenue",
            "caama_revenue",
            "caama_ir_regret",
            "caama_ex_post_ir_revenue",
        )
    }
    means_match = all(
        math.isclose(
            recomputed[key],
            float(summary[key]["mean"]),
            rel_tol=0,
            abs_tol=1e-10,
        )
        for key in recomputed
    )
    route_checks = {
        "single_preregistered_seed": config["seeds"] == [1],
        "literal_3x10_dirichlet_alpha_point_5": (
            config["distribution"] == "dirichlet_value_share"
            and config["n_bidders"] == 3
            and config["n_items"] == 10
            and config["alpha"] == 0.5
        ),
        "released_amenunet_parameterization": (
            config["parameterization"] == "amenunet_constant_context"
        ),
        "inferred_2048_menu": config["menu_size"] == 2048,
        "paper_scale_updates": (
            config["baseline_updates"] == 32_000
            and config["mutual_updates"] == 16_000
            and config["post_updates"] == 16_000
        ),
        "paper_batch_size": config["train_batch_size"] == 1024,
        "fixed_test_size": (
            config["eval_samples"] == 20_000 and len(raw) == 20_000
        ),
        "means_recomputed": means_match,
        "finite_metrics": all(math.isfinite(value) for value in recomputed.values()),
    }
    relative_errors = {
        key: abs(recomputed[key] - TARGETS[key]) / TARGETS[key]
        for key in (
            "baseline_revenue",
            "caama_revenue",
            "caama_ex_post_ir_revenue",
        )
    }
    numeric_diagnostics = {
        "relative_errors": relative_errors,
        "ir_regret_absolute_error": abs(
            recomputed["caama_ir_regret"] - TARGETS["caama_ir_regret"]
        ),
        "caama_minus_baseline": (
            recomputed["caama_revenue"] - recomputed["baseline_revenue"]
        ),
        "paper_caama_minus_baseline": (
            TARGETS["caama_revenue"] - TARGETS["baseline_revenue"]
        ),
        "one_seed_numeric_alignment_5pct": (
            max(relative_errors.values()) <= 0.05
            and abs(
                recomputed["caama_ir_regret"]
                - TARGETS["caama_ir_regret"]
            )
            <= 0.003
        ),
    }
    return {
        "raw_rows": len(raw),
        "recomputed_means": recomputed,
        "route_integrity_checks": route_checks,
        "route_integrity_pass": all(route_checks.values()),
        "numeric_diagnostics": numeric_diagnostics,
        "claim_verified": False,
        "claim_verdict": "BLOCKED",
        "blocker": "Table 1 averages five independent training seeds; this gate runs one.",
    }


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
independent = json.loads((here / "independent_checker_output.json").read_text())
config = json.loads((here / "config.json").read_text())
ok = (
    independent["route_integrity_pass"]
    and independent["raw_rows"] == 20000
    and config["baseline_updates"] == 32000
    and config["mutual_updates"] == 16000
    and config["post_updates"] == 16000
    and independent["claim_verdict"] == "BLOCKED"
    and not independent["claim_verified"]
)
print(json.dumps({
    "claim": 4,
    "route": "exact_amenunet_full_seed_1",
    "route_integrity_pass": ok,
    "verdict": "BLOCKED",
    "reason": "one of five required training seeds",
}, sort_keys=True))
sys.exit(0 if ok else 1)
"""


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
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
"""


def main() -> None:
    # Claims 1--3 are rerun by run_caama.py before this route. Claims 4 and 5
    # remain BLOCKED, with their durable evidence retained in their immutable
    # completed runs. Do not regenerate those proxy routes here: they cannot
    # change a verdict and would obscure the cost of this exact experiment.
    # A release-candidate child can aggregate the immutable run evidence after
    # this route terminates. This exact route is the only new variable.
    print(
        "CLAIM_4_CUMULATIVE_REGRESSION="
        "accepted_theory_rerun_blocked_empirical_preserved"
    )
    print(
        "CLAIM_4_PRIOR_BLOCKED_RUNS="
        "8770c5f1-7f57-4383-8caf-c69eb475714c,"
        "233ef5c5-063e-40d8-a810-ef661f153826,"
        "298cd9b9-0ac9-472b-b929-d56a9ac3613b"
    )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    started = time.perf_counter()
    raw, summary, controls, curves = et.run_seed(config, 1)
    elapsed = time.perf_counter() - started
    independent = _independent_checks(config, raw, summary)

    _write_csv(ROUTE / "raw_test_samples.csv", raw)
    _write_csv(ROUTE / "learning_curves.csv", curves)
    _write_json(ROUTE / "config.json", config)
    _write_json(ROUTE / "summary.json", summary)
    _write_json(ROUTE / "negative_control_output.json", controls)
    _write_text(
        ROUTE / "independent_checker.py", _independent_checker_source()
    )
    external_checker = subprocess.run(
        [sys.executable, str(ROUTE / "independent_checker.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    external_checker_output = (
        json.loads(external_checker.stdout)
        if external_checker.stdout.strip()
        else {"all_checks_pass": False}
    )
    independent["external_checker"] = {
        "returncode": external_checker.returncode,
        **external_checker_output,
    }
    independent["route_integrity_checks"][
        "external_independent_checker"
    ] = (
        external_checker.returncode == 0
        and external_checker_output.get("all_checks_pass") is True
    )
    independent["route_integrity_pass"] = all(
        independent["route_integrity_checks"].values()
    )
    _write_json(
        ROUTE / "claim_contract.json",
        {
            "claim": 4,
            "paper_result": TARGETS,
            "setting": "Dirichlet Value Share alpha=0.5, 3 bidders x 10 items",
            "paper_source": PAPER_URL,
            "paper_sha256": PAPER_SHA256,
            "anchors": ["S4.T1", "S5.SS1", "S5.p3", "A1.SS4"],
            "quantifiers": {
                "training_seeds": 5,
                "total_iterations_per_mechanism": 32_000,
                "test_profiles": 20_000,
            },
            "route_scope": (
                "Full paper-scale convergence gate for seed 1 only; no claim "
                "verification is possible until five-seed evidence is aggregated."
            ),
            "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
        },
    )
    _write_text(
        ROUTE / "source_audit.md",
        f"""# Claim 4 full-seed source audit

- Paper: `{PAPER_URL}`
- Paper SHA-256: `{PAPER_SHA256}`
- Released code: `{UPSTREAM_SHA}`
- Table 1 target: Randomized AMA `3.1363`, CA-AMA `3.6205`, IR regret
  `0.0031`, ex-post-IR revenue `3.5623`, averaged over five seeds.
- Section 5: 32,000 iterations, 1,024 batch size for larger settings,
  temperature 500, fixed 20,000-profile test set.
- Algorithm 1: exact argmax AMA payments in post-training.
- Unresolved source discrepancy: Table 1 says `3x10`; Table 3 and the released
  shell matrix provide only a transposed `10x3` entry. The 2,048-menu and
  initial-gamma-8 choices are therefore disclosed inferences.
""",
    )
    _write_text(
        ROUTE / "method.md",
        """# Claim 4 exact AMenuNet full-seed method

- Seed 1; literal Dirichlet Value Share(alpha=0.5), 3 bidders x 10 items.
- Released dropout-free transformer AMenuNet parameterization with 2,048 menus.
- Randomized AMA: 32,000 optimizer updates.
- CA-AMA: 16,000 mutual + 16,000 pCor-only post updates.
- Batch 1,024; softmax temperature 500; fixed seed-2002 20,000-profile test.
- Paper-stated three-linear-layer rival-only ReLU pCor MLP.
- Exact hard-argmax AMA outcome/payment in post-training and evaluation.
- Rival-profile reversal and zero-pCor identity are negative controls.
""",
    )
    _write_json(ROUTE / "independent_checker_output.json", independent)
    _write_json(
        ROUTE / "verification_criteria.json",
        {
            "route_integrity_checks": independent["route_integrity_checks"],
            "route_integrity_pass": independent["route_integrity_pass"],
            "numeric_diagnostics": independent["numeric_diagnostics"],
            "table_claim_checked": False,
            "reason": "five-seed Table-1 mean is not identified by one seed",
        },
    )
    _write_text(ROUTE / "claim_verifier.py", _claim_verifier_source())
    verifier = subprocess.run(
        [sys.executable, str(ROUTE / "claim_verifier.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_json(
        ROUTE / "claim_verifier_output.json",
        {
            "returncode": verifier.returncode,
            "stdout": verifier.stdout.strip(),
            "stderr": verifier.stderr.strip(),
        },
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
            "seed": 1,
            "elapsed_seconds": elapsed,
        },
    )
    _write_text(
        ROUTE / "limitations_and_deviations.md",
        """# Limitations and deviations

- This route completes only one of the five paper training seeds and therefore
  remains BLOCKED even if its numeric outcome aligns exactly.
- Menu size 2,048 and initial gamma 8 are inferred from the transposed 10x3
  Table-3/released-script entry; the paper never supplies a 3x10 row.
- Constant-context vectorization is algebraically regression-tested against the
  released implementation but is a CPU execution optimization.
- A divergent non-convex training result would not by itself falsify Table 1.
""",
    )
    _write_text(
        ROUTE / "EVAL.md",
        f"""# Claim 4 full-seed convergence evaluation

- Verdict: **BLOCKED** (one of five required seeds)
- Randomized AMA: `{summary['baseline_revenue']['mean']:.6f}` (paper `3.1363`)
- CA-AMA: `{summary['caama_revenue']['mean']:.6f}` (paper `3.6205`)
- IR regret: `{summary['caama_ir_regret']['mean']:.6f}` (paper `0.0031`)
- Ex-post-IR: `{summary['caama_ex_post_ir_revenue']['mean']:.6f}`
  (paper `3.5623`)
- CA-AMA minus AMA:
  `{independent['numeric_diagnostics']['caama_minus_baseline']:.6f}`
- One-seed 5% numeric alignment diagnostic:
  `{independent['numeric_diagnostics']['one_seed_numeric_alignment_5pct']}`
- Route integrity verifier exit: `{verifier.returncode}`
- Runtime: `{elapsed:.3f}` seconds.
""",
    )
    if verifier.returncode != 0:
        raise AssertionError("Claim 4 full-seed route integrity failed")
    print(
        "CLAIM_4_EXACT_FULL_SEED "
        f"seed=1 baseline={summary['baseline_revenue']['mean']:.6f} "
        f"caama={summary['caama_revenue']['mean']:.6f} "
        f"ir_regret={summary['caama_ir_regret']['mean']:.6f} "
        f"ex_post_ir={summary['caama_ex_post_ir_revenue']['mean']:.6f}"
    )
    print(
        "CLAIM_4_EXACT_FULL_SEED_NUMERIC_ALIGNMENT_5PCT="
        f"{independent['numeric_diagnostics']['one_seed_numeric_alignment_5pct']}"
    )
    print(f"CLAIM_4_EXACT_FULL_SEED_RUNTIME_SECONDS={elapsed:.6f}")
    print("CLAIM_4_EXACT_FULL_SEED_VERDICT=BLOCKED_ONE_OF_FIVE_SEEDS")


if __name__ == "__main__":
    main()
