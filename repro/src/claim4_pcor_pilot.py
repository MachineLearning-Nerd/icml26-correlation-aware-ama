#!/usr/bin/env python3
"""Multi-seed cross-item neural pCor validation for full-scale Claim 4."""
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
import torch

from empirical_train import PaperPaymentMLP


ROOT = Path(__file__).resolve().parents[2]
ROUTE = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "route_3_cross_item_pcor_multiseed"
)
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2602.09455"
PAPER_SHA256 = (
    "2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f"
)
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)
N_BIDDERS = 3
N_ITEMS = 10
ALPHA = 0.5
SEEDS = (1, 7, 19, 41, 73)
UPDATES = 2_000
BATCH_SIZE = 1_024
VALIDATION_PROFILES = 10_000
TEST_PROFILES = 20_000
BASELINE_RESERVE = 0.374
CA_CORE_RESERVE = 0.280
TARGET_IR_REGRET = 0.0031
SCALE_GRID = np.linspace(0.0, 1.0, 201)


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


def sample_torch(
    profiles: int, generator: torch.Generator
) -> torch.Tensor:
    totals = 0.5 + 0.5 * torch.rand(
        (profiles, N_ITEMS, 1), generator=generator
    )
    # _standard_gamma accepts a generator and avoids mutating global RNG state.
    concentration = torch.full(
        (profiles, N_ITEMS, N_BIDDERS), ALPHA
    )
    gamma = torch._standard_gamma(concentration, generator=generator)
    shares = gamma / gamma.sum(dim=-1, keepdim=True)
    return (totals * shares).permute(0, 2, 1).contiguous()


def reserve_core(
    values: torch.Tensor, reserve: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bidder payments, utilities, and allocated welfare."""
    sorted_values, winners = torch.sort(values, dim=1)
    top = sorted_values[:, -1]
    second = sorted_values[:, -2]
    winner = values.argmax(dim=1)
    sold = top >= reserve
    item_payment = torch.where(
        sold, torch.maximum(second, second.new_tensor(reserve)), 0.0
    )
    payment = values.new_zeros((len(values), N_BIDDERS))
    allocation_value = values.new_zeros((len(values), N_BIDDERS))
    for bidder in range(N_BIDDERS):
        won = sold & (winner == bidder)
        payment[:, bidder] = torch.where(
            won, item_payment, 0.0
        ).sum(dim=1)
        allocation_value[:, bidder] = torch.where(
            won, values[:, bidder], 0.0
        ).sum(dim=1)
    utility = allocation_value - payment
    welfare = allocation_value.sum(dim=1)
    return payment, utility, welfare


def train_pcor(
    seed: int,
) -> tuple[PaperPaymentMLP, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(51_000 + seed)
    model = PaperPaymentMLP(N_BIDDERS, N_ITEMS)
    # Zero output layers make the starting mechanism exactly the validated AMA.
    for network in model.networks:
        torch.nn.init.zeros_(network[-1].weight)
        torch.nn.init.zeros_(network[-1].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    gamma = 5.0
    target_regret = 0.001
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        learning_rate = 0.001 * min(1.0, update / 100)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        values = sample_torch(BATCH_SIZE, generator)
        with torch.no_grad():
            base_payment, base_utility, _ = reserve_core(
                values, CA_CORE_RESERVE
            )
        optimizer.zero_grad(set_to_none=True)
        extra = model(values)
        utility = base_utility - extra
        extra_revenue = extra.sum(dim=1).mean()
        regret = torch.clamp(-utility, min=0).sum(dim=1).mean()
        loss = -extra_revenue + gamma * regret
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite pCor loss at update {update}")
        loss.backward()
        optimizer.step()
        if update > 100:
            observed = max(float(regret.detach()), 1e-12)
            gamma += 0.01 * math.log(observed / target_regret)
            gamma = min(20.0, max(1.0, gamma))
        if update == 1 or update % 200 == 0 or update == UPDATES:
            row = {
                "seed": seed,
                "update": update,
                "extra_revenue": float(extra_revenue.detach()),
                "ir_regret": float(regret.detach()),
                "gamma": gamma,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(
                f"CLAIM_4_PCOR_TRAIN update={update}/{UPDATES} "
                f"seed={seed} "
                f"extra={row['extra_revenue']:.6f} "
                f"regret={row['ir_regret']:.6f} gamma={gamma:.4f}"
            )
    return model, rows


def _components(
    model: PaperPaymentMLP, values: torch.Tensor
) -> dict[str, np.ndarray]:
    arrays: dict[str, list[float]] = {
        key: [] for key in (
            "baseline_revenue",
            "ca_core_payment_0",
            "ca_core_payment_1",
            "ca_core_payment_2",
            "ca_core_utility_0",
            "ca_core_utility_1",
            "ca_core_utility_2",
            "pcor_0",
            "pcor_1",
            "pcor_2",
            "shuffled_pcor_0",
            "shuffled_pcor_1",
            "shuffled_pcor_2",
            "welfare",
        )
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 500):
            batch = values[start : start + 500]
            base_payment, _, _ = reserve_core(batch, BASELINE_RESERVE)
            ca_payment, ca_utility, welfare = reserve_core(
                batch, CA_CORE_RESERVE
            )
            extra = torch.clamp(model(batch), min=0)
            shuffled_batch = batch.flip(0)
            shuffled_extra = torch.clamp(model(shuffled_batch), min=0)
            batch_arrays = {
                "baseline_revenue": base_payment.sum(dim=1),
                "welfare": welfare,
            }
            for bidder in range(N_BIDDERS):
                batch_arrays[f"ca_core_payment_{bidder}"] = ca_payment[:, bidder]
                batch_arrays[f"ca_core_utility_{bidder}"] = ca_utility[:, bidder]
                batch_arrays[f"pcor_{bidder}"] = extra[:, bidder]
                batch_arrays[f"shuffled_pcor_{bidder}"] = shuffled_extra[:, bidder]
            for key, tensor in batch_arrays.items():
                arrays[key].extend(tensor.cpu().tolist())
    return {key: np.asarray(data) for key, data in arrays.items()}


def _metrics(
    components: dict[str, np.ndarray], scale: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ca_payment = np.column_stack(
        [components[f"ca_core_payment_{bidder}"] for bidder in range(N_BIDDERS)]
    )
    ca_utility = np.column_stack(
        [components[f"ca_core_utility_{bidder}"] for bidder in range(N_BIDDERS)]
    )
    pcor = np.column_stack(
        [components[f"pcor_{bidder}"] for bidder in range(N_BIDDERS)]
    )
    shuffled_pcor = np.column_stack(
        [components[f"shuffled_pcor_{bidder}"] for bidder in range(N_BIDDERS)]
    )
    utility = ca_utility - scale * pcor
    revenue_by_bidder = ca_payment + scale * pcor
    regret = np.maximum(-utility, 0).sum(axis=1)
    ex_post = np.where(utility >= 0, revenue_by_bidder, 0).sum(axis=1)
    shuffled_regret = np.maximum(
        -(ca_utility - scale * shuffled_pcor), 0
    ).sum(axis=1)
    raw = {
        "baseline_revenue": components["baseline_revenue"],
        "caama_revenue": revenue_by_bidder.sum(axis=1),
        "caama_ir_regret": regret,
        "caama_ex_post_ir_revenue": ex_post,
        "welfare": components["welfare"],
        "shuffled_pcor_ir_regret": shuffled_regret,
    }
    summary = {key: float(np.mean(data)) for key, data in raw.items()}
    summary["caama_minus_baseline"] = (
        summary["caama_revenue"] - summary["baseline_revenue"]
    )
    return raw, summary


def calibrate_scale(
    model: PaperPaymentMLP,
) -> tuple[float, list[dict[str, float]]]:
    values = sample_torch(
        VALIDATION_PROFILES, torch.Generator().manual_seed(88_001)
    )
    components = _components(model, values)
    rows: list[dict[str, float]] = []
    for scale in SCALE_GRID:
        _, summary = _metrics(components, float(scale))
        rows.append({
            "payment_scale": float(scale),
            "caama_revenue": summary["caama_revenue"],
            "caama_ir_regret": summary["caama_ir_regret"],
            "caama_ex_post_ir_revenue": summary[
                "caama_ex_post_ir_revenue"
            ],
        })
    feasible = [
        row for row in rows
        if row["caama_ir_regret"] <= TARGET_IR_REGRET
    ]
    if not feasible:
        raise RuntimeError("zero payment scale unexpectedly violates IR")
    selected = max(feasible, key=lambda row: row["caama_revenue"])
    return selected["payment_scale"], rows


def evaluate(
    model: PaperPaymentMLP, seed: int, scale: float
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    values = sample_torch(
        TEST_PROFILES, torch.Generator().manual_seed(99_001)
    )
    raw, summary = _metrics(_components(model, values), scale)
    rows = [
        {
            "seed": seed,
            "profile_index": index,
            "payment_scale": scale,
            **{key: float(data[index]) for key, data in raw.items()},
        }
        for index in range(TEST_PROFILES)
    ]
    summary["seed"] = seed
    summary["payment_scale"] = scale
    return rows, summary


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sem = float(scipy.stats.sem(array))
    half = float(scipy.stats.t.ppf(0.975, len(array) - 1) * sem)
    return {
        "mean": mean,
        "standard_deviation": float(array.std(ddof=1)),
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
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
"""


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
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
"""


def main() -> None:
    started = time.perf_counter()
    curves: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, float]] = []
    for seed in SEEDS:
        model, seed_curves = train_pcor(seed)
        scale, seed_calibration = calibrate_scale(model)
        seed_rows, seed_summary = evaluate(model, seed, scale)
        curves.extend(seed_curves)
        raw_rows.extend(seed_rows)
        calibration_rows.extend(
            {"seed": seed, **row} for row in seed_calibration
        )
        seed_summaries.append(seed_summary)
        print(
            "CLAIM_4_PCOR_SEED "
            f"seed={seed} scale={scale:.3f} "
            f"baseline={seed_summary['baseline_revenue']:.6f} "
            f"caama={seed_summary['caama_revenue']:.6f} "
            f"regret={seed_summary['caama_ir_regret']:.6f} "
            f"ex_post={seed_summary['caama_ex_post_ir_revenue']:.6f}"
        )
    metric_names = (
        "baseline_revenue",
        "caama_revenue",
        "caama_ir_regret",
        "caama_ex_post_ir_revenue",
        "welfare",
        "shuffled_pcor_ir_regret",
        "caama_minus_baseline",
        "payment_scale",
    )
    aggregate = {
        key: _interval([row[key] for row in seed_summaries])
        for key in metric_names
    }
    aggregate["seed_count"] = len(SEEDS)
    aggregate["seeds"] = list(SEEDS)
    aggregate["negative_control_pass"] = all(
        row["shuffled_pcor_ir_regret"] > row["caama_ir_regret"]
        for row in seed_summaries
    )
    _write_csv(ROUTE / "raw_test_profiles.csv", raw_rows)
    _write_csv(ROUTE / "learning_curve.csv", curves)
    _write_csv(ROUTE / "calibration_curve.csv", calibration_rows)
    _write_csv(ROUTE / "per_seed_summary.csv", seed_summaries)
    _write_json(ROUTE / "aggregate_summary.json", aggregate)
    _write_json(ROUTE / "claim_contract.json", {
        "claim": 4,
        "paper_result": {
            "baseline": 3.1363,
            "caama": 3.6205,
            "ir_regret": 0.0031,
            "ex_post_ir": 3.5623,
        },
        "validation_scope": {
            "training_seeds": list(SEEDS),
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "validation_profiles": VALIDATION_PROFILES,
            "test_profiles": TEST_PROFILES,
            "test_profiles_total": TEST_PROFILES * len(SEEDS),
            "payment_scale_grid_points": len(SCALE_GRID),
        },
        "verification_tolerances": {
            "revenue_relative_error": 0.05,
            "ir_regret_absolute_above_paper": 0.001,
            "paired_improvement_ci95_low_minimum": 0.0,
        },
    })
    _write_text(ROUTE / "source_audit.md", f"""# Claim 4 route 3 source audit

- Source: `{PAPER_URL}`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `{PAPER_SHA256}`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1`
- Algorithm anchor: Appendix Algorithm 1

This route uses the literal 3-bidder x 10-item Dirichlet(alpha=0.5)
distribution and the paper-stated three-linear-layer rival-only ReLU payment.
It isolates the payment learner because the public release omits a matching
3x10 checkpoint and instead lists a contradictory 10x3 configuration.
""")
    _write_text(ROUTE / "method.md", """# Cross-item pCor multi-seed method

- Fixed reserve-AMA cores come from the full-scale Route 2 optimization.
- A distinct three-layer MLP for each bidder consumes only the other bidders'
  20 values, exactly preserving DSIC.
- The pCor learner optimizes revenue plus adaptive average IR-regret penalty.
- Five seeds, 2,000 updates per seed, and batch size 1,024.
- A held-out 10,000-profile validation set selects the largest global payment
  scale with mean IR regret no greater than the paper's 0.0031. Scaling a
  rival-only payment preserves its rival-only dependence and DSIC.
- A disjoint fixed 20,000-profile hard test is used for each training seed.
- Student-t 95% intervals use training seeds as the independent units.
- Negative pCor outputs are clipped to zero at evaluation, as in the release.
- Reversing rival profiles is the negative control.
""")
    _write_json(ROUTE / "negative_control_output.json", {
        "per_seed": [
            {
                "seed": row["seed"],
                "normal_ir_regret": row["caama_ir_regret"],
                "shuffled_ir_regret": row["shuffled_pcor_ir_regret"],
                "shuffled_increases_regret": (
                    row["shuffled_pcor_ir_regret"]
                    > row["caama_ir_regret"]
                ),
            }
            for row in seed_summaries
        ],
        "all_seeds_pass": aggregate["negative_control_pass"],
    })
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
        capture_output=True,
        text=True,
        check=False,
    )
    _write_json(
        ROUTE / "claim_verifier_output.json",
        json.loads(verifier.stdout),
    )
    verdict = "VERIFIED" if verifier.returncode == 0 else "BLOCKED"
    _write_json(ROUTE / "exact_command_environment.json", {
        "fixed_command": FIXED_COMMAND,
        "git_sha": _git_sha(),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "seeds": list(SEEDS),
        "validation_seed": 88_001,
        "test_seed": 99_001,
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_text(ROUTE / "limitations_and_deviations.md", """# Limitations and deviations

- This uses 2,000 updates per seed, not the paper's 32,000-update protocol.
- The AMA core is a validated separable reserve mechanism rather than the
  unavailable learned joint 2048-menu AMenuNet checkpoint.
- The held-out scalar calibration is a DSIC-preserving constraint procedure,
  but is not described in the paper.
- A passing numeric verifier directly supports the claimed full-scale
  distribution and payment effect, but the core substitution remains a
  material fidelity limitation that must be disclosed in the final verdict.
""")
    _write_text(ROUTE / "EVAL.md", f"""# Claim 4 cross-item pCor multi-seed validation

- Machine contract: **{verdict}**
- Baseline: `{aggregate['baseline_revenue']['mean']:.6f}` (paper `3.1363`)
- CA-AMA: `{aggregate['caama_revenue']['mean']:.6f}` (paper `3.6205`)
- IR regret: `{aggregate['caama_ir_regret']['mean']:.6f}` (paper `0.0031`)
- Ex-post-IR: `{aggregate['caama_ex_post_ir_revenue']['mean']:.6f}`
  (paper `3.5623`)
- Gain 95% CI:
  `[{aggregate['caama_minus_baseline']['ci95_low']:.6f}, `
  `{aggregate['caama_minus_baseline']['ci95_high']:.6f}]`
- Independent checker exit: `{independent.returncode}`
- Claim verifier exit: `{verifier.returncode}`
""")
    print(
        "CLAIM_4_PCOR_MULTISEED "
        f"baseline={aggregate['baseline_revenue']['mean']:.6f} "
        f"caama={aggregate['caama_revenue']['mean']:.6f} "
        f"regret={aggregate['caama_ir_regret']['mean']:.6f} "
        f"ex_post={aggregate['caama_ex_post_ir_revenue']['mean']:.6f} "
        f"gain_ci95_low={aggregate['caama_minus_baseline']['ci95_low']:.6f}"
    )
    print(f"CLAIM_4_VERDICT={verdict} verifier_exit={verifier.returncode}")
    print(f"CLAIM_4_PCOR_RUNTIME_SECONDS={time.perf_counter()-started:.6f}")


if __name__ == "__main__":
    main()
