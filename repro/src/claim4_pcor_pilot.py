#!/usr/bin/env python3
"""Cross-item neural pCor pilot for the full-scale Claim 4 distribution."""
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
import torch

from empirical_train import PaperPaymentMLP


ROOT = Path(__file__).resolve().parents[2]
ROUTE = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "route_3_cross_item_pcor_pilot"
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
SEED = 1
UPDATES = 2_000
BATCH_SIZE = 1_024
TEST_PROFILES = 20_000
BASELINE_RESERVE = 0.374
CA_CORE_RESERVE = 0.280


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


def train_pcor() -> tuple[PaperPaymentMLP, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(51_001)
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
                "update": update,
                "extra_revenue": float(extra_revenue.detach()),
                "ir_regret": float(regret.detach()),
                "gamma": gamma,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(
                f"CLAIM_4_PCOR_TRAIN update={update}/{UPDATES} "
                f"extra={row['extra_revenue']:.6f} "
                f"regret={row['ir_regret']:.6f} gamma={gamma:.4f}"
            )
    return model, rows


def evaluate(model: PaperPaymentMLP) -> tuple[list[dict[str, Any]], dict[str, float]]:
    generator = torch.Generator().manual_seed(2_002)
    values = sample_torch(TEST_PROFILES, generator)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[float]] = {
        key: []
        for key in (
            "baseline_revenue",
            "caama_revenue",
            "caama_ir_regret",
            "caama_ex_post_ir_revenue",
            "welfare",
            "shuffled_pcor_ir_regret",
        )
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, TEST_PROFILES, 500):
            batch = values[start : start + 500]
            base_payment, _, _ = reserve_core(batch, BASELINE_RESERVE)
            ca_payment, ca_utility, welfare = reserve_core(
                batch, CA_CORE_RESERVE
            )
            extra = torch.clamp(model(batch), min=0)
            utility = ca_utility - extra
            ca_revenue_by_bidder = ca_payment + extra
            regret = torch.clamp(-utility, min=0).sum(dim=1)
            ex_post = torch.where(
                utility >= 0, ca_revenue_by_bidder, 0.0
            ).sum(dim=1)
            shuffled_batch = batch.flip(0)
            shuffled_extra = torch.clamp(model(shuffled_batch), min=0)
            shuffled_regret = torch.clamp(
                -(ca_utility - shuffled_extra), min=0
            ).sum(dim=1)
            batch_arrays = {
                "baseline_revenue": base_payment.sum(dim=1),
                "caama_revenue": ca_revenue_by_bidder.sum(dim=1),
                "caama_ir_regret": regret,
                "caama_ex_post_ir_revenue": ex_post,
                "welfare": welfare,
                "shuffled_pcor_ir_regret": shuffled_regret,
            }
            for key, tensor in batch_arrays.items():
                arrays[key].extend(tensor.cpu().tolist())
            for offset in range(len(batch)):
                rows.append(
                    {
                        "seed": SEED,
                        "profile_index": start + offset,
                        **{
                            key: float(tensor[offset])
                            for key, tensor in batch_arrays.items()
                        },
                    }
                )
    summary = {
        key: float(np.mean(data)) for key, data in arrays.items()
    }
    summary["caama_minus_baseline"] = (
        summary["caama_revenue"] - summary["baseline_revenue"]
    )
    return rows, summary


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
import csv
import json
import statistics
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "raw_test_profiles.csv").open()))
assert len(rows) == 20000
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
checks = {
    "row_count": len(rows) == 20000,
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


def main() -> None:
    started = time.perf_counter()
    model, curves = train_pcor()
    raw_rows, summary = evaluate(model)
    _write_csv(ROUTE / "raw_test_profiles.csv", raw_rows)
    _write_csv(ROUTE / "learning_curve.csv", curves)
    _write_json(ROUTE / "summary.json", summary)
    _write_json(ROUTE / "claim_contract.json", {
        "claim": 4,
        "paper_result": {
            "baseline": 3.1363,
            "caama": 3.6205,
            "ir_regret": 0.0031,
            "ex_post_ir": 3.5623,
        },
        "pilot_scope": {
            "training_seeds": [SEED],
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "test_profiles": TEST_PROFILES,
        },
        "status": "PILOT_ONLY_NO_VERDICT",
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
It isolates the payment learner because the public release omits the matching
3x10 core checkpoint and instead lists a contradictory 10x3 configuration.
""")
    _write_text(ROUTE / "method.md", """# Cross-item pCor pilot method

- Fixed reserve-AMA cores come from the full-scale Route 2 optimization.
- A distinct three-layer MLP for each bidder consumes only the other bidders'
  20 values, exactly preserving DSIC.
- The pCor learner optimizes revenue plus adaptive average IR-regret penalty.
- One seed, 2,000 updates, batch 1,024, and a fixed 20,000-profile hard test.
- Negative pCor outputs are clipped to zero at evaluation, as in the release.
- Reversing rival profiles is the negative control.
""")
    _write_json(ROUTE / "negative_control_output.json", {
        "normal_ir_regret": summary["caama_ir_regret"],
        "shuffled_ir_regret": summary["shuffled_pcor_ir_regret"],
        "shuffled_increases_regret": (
            summary["shuffled_pcor_ir_regret"]
            > summary["caama_ir_regret"]
        ),
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
        "seed": SEED,
        "test_seed": 2_002,
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_text(ROUTE / "limitations_and_deviations.md", """# Limitations and deviations

- One training seed and 2,000 updates are a throughput/optimization pilot, not
  the paper's five-seed, 32,000-update protocol.
- The AMA core is a validated separable reserve mechanism rather than the
  unavailable learned joint 2048-menu AMenuNet checkpoint.
- A promising numeric result would justify a full child; it would not by
  itself verify Claim 4.
""")
    _write_text(ROUTE / "EVAL.md", f"""# Claim 4 cross-item pCor pilot

- Status: **PILOT ONLY — NO CLAIM VERDICT**
- Baseline: `{summary['baseline_revenue']:.6f}` (paper `3.1363`)
- CA-AMA: `{summary['caama_revenue']:.6f}` (paper `3.6205`)
- IR regret: `{summary['caama_ir_regret']:.6f}` (paper `0.0031`)
- Ex-post-IR: `{summary['caama_ex_post_ir_revenue']:.6f}` (paper `3.5623`)
- Shuffled-rival regret: `{summary['shuffled_pcor_ir_regret']:.6f}`
- Independent checker exit: `{independent.returncode}`
""")
    print(
        "CLAIM_4_PCOR_PILOT "
        f"baseline={summary['baseline_revenue']:.6f} "
        f"caama={summary['caama_revenue']:.6f} "
        f"regret={summary['caama_ir_regret']:.6f} "
        f"ex_post={summary['caama_ex_post_ir_revenue']:.6f}"
    )
    print("CLAIM_4_VERDICT=PILOT_ONLY_NO_VERDICT")
    print(f"CLAIM_4_PCOR_RUNTIME_SECONDS={time.perf_counter()-started:.6f}")


if __name__ == "__main__":
    main()
