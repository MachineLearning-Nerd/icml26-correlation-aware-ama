#!/usr/bin/env python3
"""Vectorized CPU training route for the paper's Table-1 AMA experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "repro" / "config" / "empirical_train.json"
ARTIFACT_ROOT = ROOT / ".openresearch" / "artifacts"
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2602.09455"
PAPER_SHA256 = "2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f"
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _upstream_classes():
    upstream = str(ROOT / "upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from net import TransformerMechanism

    return TransformerMechanism


class EfficientAMA(nn.Module):
    """AMenuNet mechanism with constant contexts evaluated once per update.

    The released implementation repeats identical bidder/item identifiers across
    the batch before applying a dropout-free network. Evaluating those identifiers
    once yields identical allocations, weights, and boosts while eliminating
    redundant Transformer work. Welfare, soft choice, and pivot payments remain
    per valuation sample.
    """

    def __init__(
        self,
        n_bidders: int,
        n_items: int,
        menu_size: int,
        allocation_temperature: float = 10.0,
    ) -> None:
        super().__init__()
        transformer = _upstream_classes()
        self.n_bidders = n_bidders
        self.n_items = n_items
        self.menu_size = menu_size
        self.allocation_temperature = allocation_temperature
        self.mechanism = transformer(
            n_bidders + 1,
            n_items,
            10,
            3,
            4,
            64,
            menu_size,
            continuous_context=False,
            cond_prob=False,
        )
        self.register_buffer(
            "bidder_context", torch.arange(n_bidders + 1).reshape(1, -1).long()
        )
        self.register_buffer(
            "item_context", torch.arange(n_items).reshape(1, -1).long()
        )

    def parameters_for_auction(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        allocations, weights, boosts = self.mechanism(
            (self.bidder_context, self.item_context),
            self.allocation_temperature,
        )
        allocations = allocations[0]
        weights = weights[0]
        boosts = boosts[0]
        allocations = torch.cat(
            (
                allocations,
                torch.zeros(
                    1,
                    self.n_bidders,
                    self.n_items,
                    dtype=allocations.dtype,
                    device=allocations.device,
                ),
            ),
            dim=0,
        )
        boosts = torch.cat((boosts, boosts.new_zeros(1)))
        return allocations, weights, boosts

    @staticmethod
    def _welfare(
        values: torch.Tensor,
        allocations: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        item_values = torch.einsum("tnm,bnm->btn", allocations, values)
        return item_values * weights.reshape(1, 1, -1)

    def soft_outcomes(
        self, values: torch.Tensor, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        allocations, weights, boosts = self.parameters_for_auction()
        welfare = self._welfare(values, allocations, weights)
        scores = welfare.sum(dim=-1) + boosts
        choice = torch.softmax(scores * temperature, dim=-1)
        chosen_welfare = (welfare * choice.unsqueeze(-1)).sum(dim=1)
        chosen_boost = (choice * boosts).sum(dim=1)
        payments: list[torch.Tensor] = []
        for bidder in range(self.n_bidders):
            removed_welfare = welfare.sum(dim=-1) - welfare[:, :, bidder]
            removed_choice = torch.softmax(
                (removed_welfare + boosts) * temperature, dim=-1
            )
            removed_value = (
                removed_choice * (removed_welfare + boosts)
            ).sum(dim=1)
            current_others = (
                chosen_welfare.sum(dim=-1)
                - chosen_welfare[:, bidder]
                + chosen_boost
            )
            payments.append(
                (removed_value - current_others) / weights[bidder]
            )
        payment = torch.stack(payments, dim=1)
        valuation = chosen_welfare / weights.reshape(1, -1)
        return payment, valuation, choice

    def hard_outcomes(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        allocations, weights, boosts = self.parameters_for_auction()
        welfare = self._welfare(values, allocations, weights)
        scores = welfare.sum(dim=-1) + boosts
        choice_index = torch.argmax(scores, dim=-1)
        rows = torch.arange(values.shape[0], device=values.device)
        chosen_welfare = welfare[rows, choice_index]
        chosen_boost = boosts[choice_index]
        payments: list[torch.Tensor] = []
        for bidder in range(self.n_bidders):
            removed_welfare = welfare.sum(dim=-1) - welfare[:, :, bidder]
            removed_scores = removed_welfare + boosts
            removed_value = torch.max(removed_scores, dim=-1).values
            current_others = (
                chosen_welfare.sum(dim=-1)
                - chosen_welfare[:, bidder]
                + chosen_boost
            )
            payments.append(
                (removed_value - current_others) / weights[bidder]
            )
        payment = torch.stack(payments, dim=1)
        valuation = chosen_welfare / weights.reshape(1, -1)
        return payment, valuation, choice_index


class PaperPaymentMLP(nn.Module):
    """Three-linear-layer ReLU pCor network stated in Section 5."""

    def __init__(self, n_bidders: int, n_items: int) -> None:
        super().__init__()
        input_dim = (n_bidders - 1) * n_items
        hidden = 8 * input_dim
        self.n_bidders = n_bidders
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(n_bidders)
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        outputs = []
        for bidder, network in enumerate(self.networks):
            rivals = torch.cat(
                (values[:, :bidder], values[:, bidder + 1 :]), dim=1
            ).flatten(1)
            outputs.append(network(rivals).squeeze(-1))
        return torch.stack(outputs, dim=1)


def sample_values(
    config: dict[str, Any], batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    n = int(config["n_bidders"])
    m = int(config["n_items"])
    alpha = float(config["alpha"])
    distribution = config["distribution"]
    if distribution == "paper_literal_bernoulli_mixture":
        if n != 2:
            raise ValueError("linear asymmetric distribution requires two bidders")
        v1 = torch.rand((batch_size, m), generator=generator)
        dependent = torch.rand((batch_size, m), generator=generator) < alpha
        independent = torch.rand((batch_size, m), generator=generator) / 4.0
        v2 = torch.where(dependent, (1.0 - v1) / 4.0, independent)
        return torch.stack((v1, v2), dim=1)
    if distribution == "dirichlet_value_share":
        totals = 0.5 + 0.5 * torch.rand(
            (batch_size, m), generator=generator
        )
        # torch.distributions does not accept a Generator, so isolate its global
        # RNG with the seed established once per training seed.
        shares = torch.distributions.Dirichlet(
            torch.full((n,), alpha)
        ).sample((batch_size, m))
        return (shares * totals.unsqueeze(-1)).permute(0, 2, 1)
    raise ValueError(f"unknown distribution: {distribution}")


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _learning_rate(config: dict[str, Any], update: int) -> float:
    final_lr = float(config["learning_rate"])
    warmup = int(config["warmup_updates"])
    return final_lr * min(1.0, update / max(1, warmup))


def train_baseline(
    config: dict[str, Any],
    seed: int,
    curve_rows: list[dict[str, Any]],
) -> EfficientAMA:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed * 1009 + 17)
    model = EfficientAMA(
        int(config["n_bidders"]),
        int(config["n_items"]),
        int(config["menu_size"]),
        float(config["allocation_temperature"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    started = time.perf_counter()
    total = int(config["baseline_updates"])
    for update in range(1, total + 1):
        _set_lr(optimizer, _learning_rate(config, update))
        values = sample_values(
            config, int(config["train_batch_size"]), generator
        )
        optimizer.zero_grad(set_to_none=True)
        payment, _, _ = model.soft_outcomes(
            values, float(config["softmax_temperature"])
        )
        loss = -payment.sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"baseline loss became nonfinite at {update}")
        loss.backward()
        optimizer.step()
        if update == 1 or update % int(config["log_every"]) == 0 or update == total:
            row = {
                "seed": seed,
                "phase": "baseline",
                "update": update,
                "revenue": float((-loss).detach()),
                "ir_regret": "",
                "gamma": "",
                "elapsed_seconds": time.perf_counter() - started,
            }
            curve_rows.append(row)
            print(
                f"TRAIN seed={seed} phase=baseline update={update}/{total} "
                f"revenue={row['revenue']:.6f}"
            )
    return model


def train_caama(
    config: dict[str, Any],
    seed: int,
    curve_rows: list[dict[str, Any]],
) -> tuple[EfficientAMA, PaperPaymentMLP]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed * 1013 + 29)
    model = EfficientAMA(
        int(config["n_bidders"]),
        int(config["n_items"]),
        int(config["menu_size"]),
        float(config["allocation_temperature"]),
    )
    pcor = PaperPaymentMLP(
        int(config["n_bidders"]), int(config["n_items"])
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pcor.parameters()), lr=1e-8
    )
    gamma = float(config["gamma_initial"])
    mutual = int(config["mutual_updates"])
    post = int(config["post_updates"])
    target = float(config["target_ir_regret"])
    started = time.perf_counter()

    def update_once(update: int, phase: str) -> tuple[float, float]:
        nonlocal gamma
        _set_lr(optimizer, _learning_rate(config, update))
        values = sample_values(
            config, int(config["train_batch_size"]), generator
        )
        optimizer.zero_grad(set_to_none=True)
        if phase == "post":
            with torch.no_grad():
                payment, valuation, _ = model.soft_outcomes(
                    values, float(config["softmax_temperature"])
                )
        else:
            payment, valuation, _ = model.soft_outcomes(
                values, float(config["softmax_temperature"])
            )
        extra = pcor(values)
        utility = valuation - payment - extra
        revenue = (payment + extra).sum(dim=1).mean()
        regret = torch.clamp(-utility, min=0).sum(dim=1).mean()
        loss = -revenue + gamma * regret
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"CA-AMA loss became nonfinite at {phase} update {update}"
            )
        loss.backward()
        optimizer.step()
        if update > int(config["warmup_updates"]):
            observed = max(float(regret.detach()), 1e-12)
            gamma += float(config["gamma_learning_rate"]) * math.log(
                observed / target
            )
            gamma = min(
                float(config["gamma_max"]),
                max(float(config["gamma_min"]), gamma),
            )
        return float(revenue.detach()), float(regret.detach())

    for update in range(1, mutual + 1):
        revenue, regret = update_once(update, "mutual")
        if update == 1 or update % int(config["log_every"]) == 0 or update == mutual:
            curve_rows.append(
                {
                    "seed": seed,
                    "phase": "mutual",
                    "update": update,
                    "revenue": revenue,
                    "ir_regret": regret,
                    "gamma": gamma,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"TRAIN seed={seed} phase=mutual update={update}/{mutual} "
                f"revenue={revenue:.6f} regret={regret:.6f} gamma={gamma:.4f}"
            )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(pcor.parameters(), lr=1e-8)
    for update in range(1, post + 1):
        revenue, regret = update_once(update, "post")
        if update == 1 or update % int(config["log_every"]) == 0 or update == post:
            curve_rows.append(
                {
                    "seed": seed,
                    "phase": "post",
                    "update": update,
                    "revenue": revenue,
                    "ir_regret": regret,
                    "gamma": gamma,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"TRAIN seed={seed} phase=post update={update}/{post} "
                f"revenue={revenue:.6f} regret={regret:.6f} gamma={gamma:.4f}"
            )
    return model, pcor


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(count)
    critical = float(scipy.stats.t.ppf(0.975, count - 1))
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "standard_error": se,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
    }


def evaluate_models(
    config: dict[str, Any],
    seed: int,
    baseline: EfficientAMA,
    ca_model: EfficientAMA,
    pcor: PaperPaymentMLP,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generator = torch.Generator().manual_seed(2002)
    test_values = sample_values(config, int(config["eval_samples"]), generator)
    raw_rows: list[dict[str, Any]] = []
    shuffled_rows: list[float] = []
    batch_size = int(config["eval_batch_size"])
    baseline.eval()
    ca_model.eval()
    pcor.eval()
    with torch.no_grad():
        for start in range(0, len(test_values), batch_size):
            values = test_values[start : start + batch_size]
            base_payment, _, _ = baseline.hard_outcomes(values)
            ca_payment, ca_valuation, _ = ca_model.hard_outcomes(values)
            # Match the released evaluator: negative learned additions are
            # truncated to zero before revenue and IR accounting.
            extra = torch.clamp(pcor(values), min=0)
            utility = ca_valuation - ca_payment - extra
            permutation = torch.arange(len(values) - 1, -1, -1)
            shuffled_values = values[permutation]
            shuffled_extra = torch.clamp(pcor(shuffled_values), min=0)
            shuffled_utility = ca_valuation - ca_payment - shuffled_extra
            shuffled_rows.extend(
                (ca_payment + shuffled_extra).sum(dim=1).cpu().tolist()
            )
            base_revenue = base_payment.sum(dim=1)
            ca_revenue = (ca_payment + extra).sum(dim=1)
            regret = torch.clamp(-utility, min=0).sum(dim=1)
            valid_revenue = torch.where(
                utility >= 0, ca_payment + extra, torch.zeros_like(ca_payment)
            ).sum(dim=1)
            shuffled_regret = torch.clamp(-shuffled_utility, min=0).sum(dim=1)
            for offset in range(len(values)):
                raw_rows.append(
                    {
                        "seed": seed,
                        "sample_index": start + offset,
                        "baseline_revenue": float(base_revenue[offset]),
                        "caama_revenue": float(ca_revenue[offset]),
                        "caama_ir_regret": float(regret[offset]),
                        "caama_ex_post_ir_revenue": float(valid_revenue[offset]),
                        "zero_pcor_revenue": float(ca_payment[offset].sum()),
                        "shuffled_pcor_revenue": float(
                            (ca_payment[offset] + shuffled_extra[offset]).sum()
                        ),
                        "shuffled_pcor_ir_regret": float(shuffled_regret[offset]),
                    }
                )
    summary: dict[str, Any] = {}
    for key in (
        "baseline_revenue",
        "caama_revenue",
        "caama_ir_regret",
        "caama_ex_post_ir_revenue",
        "zero_pcor_revenue",
    ):
        summary[key] = _metric_summary(
            np.asarray([row[key] for row in raw_rows], dtype=float)
        )
    controls = {
        "rival_profile_reversal": {
            "purpose": (
                "break the samplewise rival/own-bid relationship while leaving "
                "the trained mechanism fixed"
            ),
            "revenue": _metric_summary(np.asarray(shuffled_rows, dtype=float)),
            "ir_regret": _metric_summary(
                np.asarray(
                    [row["shuffled_pcor_ir_regret"] for row in raw_rows],
                    dtype=float,
                )
            ),
        },
        "zero_pcor_identity": {
            "mean_zero_pcor_revenue": summary["zero_pcor_revenue"]["mean"],
            "interpretation": (
                "With pCor identically zero, the CA model reduces to its AMA component."
            ),
        },
    }
    return raw_rows, summary, controls


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(
    config: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    curve_rows: list[dict[str, Any]] = []
    baseline = train_baseline(config, seed, curve_rows)
    ca_model, pcor = train_caama(config, seed, curve_rows)
    raw, summary, controls = evaluate_models(
        config, seed, baseline, ca_model, pcor
    )
    return raw, summary, controls, curve_rows


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["mode"] != "pilot_no_verdict":
        raise ValueError(
            "This branch is deliberately gated to pilot_no_verdict; "
            "a full-evidence child must change the committed mode."
        )
    started = time.perf_counter()
    claim = int(config["claim"])
    route = ARTIFACT_ROOT / f"claim_{claim}" / "route_2_vectorized_pilot"
    all_raw: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    for seed in config["seeds"]:
        raw, summary, control, curves = run_seed(config, int(seed))
        all_raw.extend(raw)
        all_curves.extend(curves)
        summaries[str(seed)] = summary
        controls[str(seed)] = control
        print(
            f"PILOT_RESULT seed={seed} "
            f"baseline={summary['baseline_revenue']['mean']:.6f} "
            f"caama={summary['caama_revenue']['mean']:.6f} "
            f"ir_regret={summary['caama_ir_regret']['mean']:.6f} "
            f"ex_post_ir={summary['caama_ex_post_ir_revenue']['mean']:.6f}"
        )
    _write_csv(route / "raw_test_samples.csv", all_raw)
    _write_csv(route / "learning_curves.csv", all_curves)
    _write_json(route / "summary.json", summaries)
    _write_json(route / "negative_control_output.json", controls)
    _write_json(route / "config.json", config)
    _write_json(
        route / "exact_command_environment.json",
        {
            "fixed_command": FIXED_COMMAND,
            "git_sha": _git_sha(),
            "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _write_text(
        route / "method.md",
        """# Route 2 method

This route interprets the paper's 32,000 iterations as 32,000 minibatch
updates, uses the literal Section 5.1 Bernoulli mixture, and uses the stated
three-layer ReLU correlation-payment network. AMenuNet's constant bidder/item
contexts are evaluated once per update; the resulting allocation menu,
weights, boosts, welfare, and pivot-payment equations are unchanged.

This child is a 1,000-update, one-seed end-to-end pilot. It is capacity and
correctness evidence only, not evidence for the reported Table-1 number.
""",
    )
    _write_text(
        route / "STATUS.md",
        """# Route 2 status

- Status: **PILOT ONLY — NO CLAIM VERDICT**
- Full paper scale: not yet run.
- Purpose: verify vectorization, optimization, hard-payment evaluation,
  uncertainty calculation, IR accounting, and negative controls.
""",
    )
    print(f"EMPIRICAL_TRAIN_MODE={config['mode']}")
    print("CLAIM_5_STATUS=PILOT_ONLY_NO_VERDICT")
    print(f"EMPIRICAL_TRAIN_RUNTIME_SECONDS={time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
