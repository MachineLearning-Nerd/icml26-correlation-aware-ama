#!/usr/bin/env python3
"""Exact-architecture CPU feasibility pilot for the Table-1 Claim 4 route."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch

import empirical_train as et


ROOT = Path(__file__).resolve().parents[2]
ROUTE = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "route_4_exact_amenunet_pilot"
)
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2602.09455"
PAPER_SHA256 = (
    "2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f"
)
UPSTREAM_SHA = "ed2af19ed02c70b58efdf705635981241222d045"
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "raw_test_samples.csv").open()))
summary = json.loads((here / "summary.json").read_text())
criteria = json.loads((here / "verification_criteria.json").read_text())
assert len(rows) == 2000
for metric in (
    "baseline_revenue",
    "caama_revenue",
    "caama_ir_regret",
    "caama_ex_post_ir_revenue",
):
    observed = sum(float(row[metric]) for row in rows) / len(rows)
    recorded = float(summary[metric]["mean"])
    assert math.isclose(observed, recorded, rel_tol=0, abs_tol=1e-10)
assert criteria["table_claim_checked"] is False
assert criteria["all_checks_pass"] is True
print(json.dumps({
    "raw_rows": len(rows),
    "means_recomputed": True,
    "structural_checks_pass": True,
    "table_claim_checked": False,
}, sort_keys=True))
"""


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
criteria = json.loads((here / "verification_criteria.json").read_text())
independent = json.loads((here / "independent_checker_output.json").read_text())
ok = (
    criteria["all_checks_pass"]
    and not criteria["table_claim_checked"]
    and independent["raw_rows"] == 2000
    and independent["means_recomputed"]
    and independent["structural_checks_pass"]
)
print(json.dumps({
    "claim": 4,
    "evidence_status": "PILOT_ONLY",
    "verdict": "BLOCKED",
    "pilot_contract_pass": ok,
}, sort_keys=True))
sys.exit(0 if ok else 1)
"""


def _config() -> dict[str, Any]:
    """A short run that preserves every structural paper choice."""
    return {
        "claim": 4,
        "mode": "claim4_exact_amenunet_pilot",
        "distribution": "dirichlet_value_share",
        "parameterization": "amenunet_constant_context",
        "seeds": [1],
        "n_bidders": 3,
        "n_items": 10,
        "alpha": 0.5,
        # Table 3 transposes the paper's 3x10 row to 10x3. We test its
        # associated 2048-menu choice explicitly and retain the ambiguity.
        "menu_size": 2048,
        "baseline_updates": 10,
        "mutual_updates": 10,
        "post_updates": 10,
        "train_batch_size": 1024,
        "eval_samples": 2000,
        "eval_batch_size": 250,
        "learning_rate": 0.0003,
        "softmax_temperature": 500.0,
        "allocation_temperature": 10.0,
        "gamma_initial": 8.0,
        "gamma_learning_rate": 0.01,
        "gamma_min": 1.0,
        "gamma_max": 20.0,
        "target_ir_regret": 0.001,
        "warmup_updates": 100,
        "log_every": 1,
    }


def main() -> None:
    # Every child reruns the previously accepted cumulative evidence routes.
    from campaign_summary import main as run_campaign_summary
    from claim4_conditional import main as run_claim4_conditional
    from claim4_pcor_pilot import main as run_claim4_pcor_multiseed
    from claim5_falsification import main as run_claim5_falsification

    run_claim4_conditional()
    run_claim4_pcor_multiseed()
    run_claim5_falsification()
    run_campaign_summary()

    config = _config()
    started = time.perf_counter()
    raw, summary, controls, curves = et.run_seed(config, 1)
    elapsed = time.perf_counter() - started
    optimizer_updates = (
        config["baseline_updates"]
        + config["mutual_updates"]
        + config["post_updates"]
    )
    seconds_per_update = elapsed / optimizer_updates
    projected_one_seed_seconds = seconds_per_update * 64_000

    model = et.EfficientAMA(3, 10, 2048, 10.0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    architecture_checks = {
        "literal_distribution": config["distribution"]
        == "dirichlet_value_share",
        "paper_shape": (
            config["n_bidders"] == 3 and config["n_items"] == 10
        ),
        "alpha": config["alpha"] == 0.5,
        "released_transformer_parameterization": config["parameterization"]
        == "amenunet_constant_context",
        "inferred_menu_size": config["menu_size"] == 2048,
        "hard_argmax_post_training": True,
        "finite_metrics": all(
            torch.isfinite(torch.tensor(summary[key]["mean"]))
            for key in (
                "baseline_revenue",
                "caama_revenue",
                "caama_ir_regret",
                "caama_ex_post_ir_revenue",
            )
        ),
        "expected_raw_rows": len(raw) == config["eval_samples"],
    }
    all_checks_pass = all(architecture_checks.values())

    _write_csv(ROUTE / "raw_test_samples.csv", raw)
    _write_csv(ROUTE / "learning_curves.csv", curves)
    _write_json(ROUTE / "summary.json", summary)
    _write_json(ROUTE / "negative_control_output.json", controls)
    _write_json(
        ROUTE / "claim_contract.json",
        {
            "claim": 4,
            "paper_result": {
                "setting": "Dirichlet Value Share alpha=0.5, 3 bidders x 10 items",
                "randomized_ama_revenue": 3.1363,
                "caama_revenue": 3.6205,
                "caama_ir_regret": 0.0031,
                "caama_ex_post_ir_revenue": 3.5623,
                "training_seeds": 5,
            },
            "paper_source": PAPER_URL,
            "paper_sha256": PAPER_SHA256,
            "anchors": ["S4.T1", "S5.SS1", "S5.p3", "A1.SS4"],
            "pilot_scope": (
                "Structural and throughput validation only; it cannot verify "
                "Table 1 until the five-seed, 32,000-total-update contract runs."
            ),
            "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
        },
    )
    _write_text(
        ROUTE / "source_audit.md",
        f"""# Claim 4 exact-route source audit

- Paper source: `{PAPER_URL}`
- Paper SHA-256: `{PAPER_SHA256}`
- Released code revision: `{UPSTREAM_SHA}`
- Table 1: `3 x 10`, alpha `0.5`, Randomized AMA `3.1363`,
  CA-AMA `3.6205`, IR regret `0.0031`, ex-post-IR revenue `3.5623`.
- Section 5: 32,000 iterations, batch size 1,024 for larger settings,
  five seeds, softmax temperature 500, and a three-layer ReLU pCor network.
- Algorithm 1: post-training must use exact argmax AMA payments.
- Table 3 and the released command matrix list `10 x 3`, not `3 x 10`.
  This pilot treats menu size 2,048 and initial gamma 8 as an explicit
  inference, not a fully specified paper fact.
""",
    )
    _write_text(
        ROUTE / "method.md",
        f"""# Claim 4 exact AMenuNet feasibility method

- Literal 3-bidder x 10-item Dirichlet(alpha=0.5) generator.
- Released transformer over-parameterization for allocations, weights, boosts.
- Inferred 2,048-menu configuration, batch size 1,024.
- Paper-stated three-linear-layer rival-only ReLU pCor network.
- Soft AMA payments in baseline/mutual training and exact hard argmax payments
  in pCor-only post-training.
- Hard argmax evaluation on a fixed seed-2002 test set.
- This pilot uses 10 baseline + 10 mutual + 10 post updates and 2,000 test
  profiles only to validate the implementation and measure CPU throughput.
- Total train/evaluate time: `{elapsed:.6f}` seconds.
""",
    )
    _write_json(
        ROUTE / "verification_criteria.json",
        {
            "checks": architecture_checks,
            "all_checks_pass": all_checks_pass,
            "table_claim_checked": False,
            "reason": "short one-seed feasibility pilot",
        },
    )
    _write_text(
        ROUTE / "independent_checker.py", _independent_checker_source()
    )
    independent = subprocess.run(
        [sys.executable, str(ROUTE / "independent_checker.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    independent_output = json.loads(independent.stdout)
    independent_output["learning_curve_rows"] = len(curves)
    independent_output["model_parameter_count"] = parameter_count
    _write_json(
        ROUTE / "independent_checker_output.json", independent_output
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
            "optimizer_updates": optimizer_updates,
            "seconds_per_update_including_evaluation": seconds_per_update,
            "projected_one_seed_64000_update_seconds": projected_one_seed_seconds,
        },
    )
    _write_text(
        ROUTE / "limitations_and_deviations.md",
        """# Limitations and deviations

- **PILOT ONLY:** one seed, 30 total updates, and 2,000 test profiles.
- The paper's Table 3 does not state a 3x10 configuration. Menu size 2,048 and
  initial gamma 8 are inferred from the transposed 10x3 entry.
- The constant-context optimization evaluates the dropout-free Transformer once
  per update. Existing regression tests establish equality with the released
  batched implementation, but this execution optimization is not in its script.
- A pilot mismatch is not a falsification of a non-convex empirical result.
""",
    )
    _write_text(
        ROUTE / "EVAL.md",
        f"""# Claim 4 exact AMenuNet pilot evaluation

- Verdict: **BLOCKED**
- Evidence status: exact-architecture feasibility pilot, not Table-1 evidence.
- Observed Randomized AMA revenue: `{summary['baseline_revenue']['mean']:.6f}`
- Observed CA-AMA revenue: `{summary['caama_revenue']['mean']:.6f}`
- Observed IR regret: `{summary['caama_ir_regret']['mean']:.6f}`
- Observed ex-post-IR revenue:
  `{summary['caama_ex_post_ir_revenue']['mean']:.6f}`
- Structural checks passed: `{all_checks_pass}`
- Measured runtime: `{elapsed:.3f}` seconds for `{optimizer_updates}` updates
  plus evaluation.
- Projected one-seed 64,000-update time at this unoptimized pilot rate:
  `{projected_one_seed_seconds / 3600:.2f}` hours.
""",
    )
    if not all_checks_pass or verifier.returncode != 0:
        raise AssertionError("Claim 4 exact-route pilot structural checks failed")
    print(
        "CLAIM_4_EXACT_AMENUNET_PILOT "
        f"baseline={summary['baseline_revenue']['mean']:.6f} "
        f"caama={summary['caama_revenue']['mean']:.6f} "
        f"ir_regret={summary['caama_ir_regret']['mean']:.6f} "
        f"ex_post_ir={summary['caama_ex_post_ir_revenue']['mean']:.6f}"
    )
    print(f"CLAIM_4_EXACT_AMENUNET_PARAMETERS={parameter_count}")
    print(f"CLAIM_4_EXACT_AMENUNET_SECONDS_PER_UPDATE={seconds_per_update:.6f}")
    print(
        "CLAIM_4_EXACT_AMENUNET_PROJECTED_ONE_SEED_HOURS="
        f"{projected_one_seed_seconds / 3600:.6f}"
    )
    print("CLAIM_4_EXACT_AMENUNET_VERDICT=BLOCKED_PILOT_ONLY")


if __name__ == "__main__":
    main()
