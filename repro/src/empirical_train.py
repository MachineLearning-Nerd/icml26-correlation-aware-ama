#!/usr/bin/env python3
"""Vectorized CPU training route for the paper's Table-1 AMA experiments."""
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


class DirectAMA(EfficientAMA):
    """The same AMA mechanism space with allocations/weights/boosts explicit."""

    def __init__(
        self,
        n_bidders: int,
        n_items: int,
        menu_size: int,
        allocation_temperature: float = 10.0,
    ) -> None:
        nn.Module.__init__(self)
        self.n_bidders = n_bidders
        self.n_items = n_items
        self.menu_size = menu_size
        self.allocation_temperature = allocation_temperature
        self.allocation_logits = nn.Parameter(
            torch.randn(menu_size, n_bidders + 1, n_items) * 0.01
        )
        self.weight_logits = nn.Parameter(torch.zeros(n_bidders))
        self.boosts = nn.Parameter(torch.randn(menu_size) * 0.01)

    def parameters_for_auction(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feasible = torch.softmax(
            self.allocation_logits * self.allocation_temperature, dim=1
        )
        allocations = feasible[:, : self.n_bidders]
        weights = torch.sigmoid(self.weight_logits)
        allocations = torch.cat(
            (
                allocations,
                allocations.new_zeros(
                    1, self.n_bidders, self.n_items
                ),
            ),
            dim=0,
        )
        boosts = torch.cat((self.boosts, self.boosts.new_zeros(1)))
        return allocations, weights, boosts


def make_ama(config: dict[str, Any]) -> EfficientAMA:
    parameters = (
        int(config["n_bidders"]),
        int(config["n_items"]),
        int(config["menu_size"]),
        float(config["allocation_temperature"]),
    )
    if config.get("parameterization", "amenunet_constant_context") == (
        "direct_mechanism_space"
    ):
        return DirectAMA(*parameters)
    return EfficientAMA(*parameters)


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
        # torch.distributions.Dirichlet.sample has no Generator argument.
        # Sampling its gamma representation directly keeps both training and
        # the fixed seed-2002 test set scoped to the supplied generator.
        concentration = torch.full((batch_size, m, n), alpha)
        gamma = torch._standard_gamma(
            concentration, generator=generator
        )
        shares = gamma / gamma.sum(dim=-1, keepdim=True)
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
    model = make_ama(config)
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
    model = make_ama(config)
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
                # Algorithm 1, lines 10--13: post-training freezes the AMA
                # parameters and uses the true argmax AMA payment/utility.
                payment, valuation, _ = model.hard_outcomes(values)
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


def aggregate_seed_results(summaries: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "baseline_revenue",
        "caama_revenue",
        "caama_ir_regret",
        "caama_ex_post_ir_revenue",
        "zero_pcor_revenue",
    )
    aggregate: dict[str, Any] = {}
    ordered_seeds = sorted(summaries, key=int)
    for metric in metrics:
        seed_means = np.asarray(
            [summaries[seed][metric]["mean"] for seed in ordered_seeds],
            dtype=float,
        )
        aggregate[metric] = {
            **_metric_summary(seed_means),
            "seed_means": seed_means.tolist(),
        }
    improvement = np.asarray(
        [
            summaries[seed]["caama_revenue"]["mean"]
            - summaries[seed]["baseline_revenue"]["mean"]
            for seed in ordered_seeds
        ],
        dtype=float,
    )
    aggregate["paired_caama_minus_baseline"] = {
        **_metric_summary(improvement),
        "seed_differences": improvement.tolist(),
    }
    return aggregate


def verification_criteria(
    config: dict[str, Any],
    aggregate: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    targets = {
        "baseline_revenue": 1.7135,
        "caama_revenue": 1.9359,
        "caama_ir_regret": 0.0052,
        "caama_ex_post_ir_revenue": 1.8553,
    }
    tolerance = config["verification_tolerances"]
    relative_errors = {
        key: abs(aggregate[key]["mean"] - target) / abs(target)
        for key, target in targets.items()
        if key != "caama_ir_regret"
    }
    regret_error = abs(
        aggregate["caama_ir_regret"]["mean"]
        - targets["caama_ir_regret"]
    )
    normal_regrets = np.asarray(
        [
            aggregate_value
            for aggregate_value in aggregate["caama_ir_regret"]["seed_means"]
        ],
        dtype=float,
    )
    shuffled_regrets = np.asarray(
        [
            controls[seed]["rival_profile_reversal"]["ir_regret"]["mean"]
            for seed in sorted(controls, key=int)
        ],
        dtype=float,
    )
    shuffled_effect = _metric_summary(shuffled_regrets - normal_regrets)
    checks = {
        "five_training_seeds": len(config["seeds"]) == 5,
        "paper_scale_updates": (
            int(config["baseline_updates"]) == 32_000
            and int(config["mutual_updates"]) + int(config["post_updates"])
            == 32_000
        ),
        "fixed_test_size": int(config["eval_samples"]) == 20_000,
        "baseline_within_preregistered_tolerance": (
            relative_errors["baseline_revenue"]
            <= float(tolerance["revenue_relative_error"])
        ),
        "caama_within_preregistered_tolerance": (
            relative_errors["caama_revenue"]
            <= float(tolerance["revenue_relative_error"])
        ),
        "ex_post_ir_within_preregistered_tolerance": (
            relative_errors["caama_ex_post_ir_revenue"]
            <= float(tolerance["revenue_relative_error"])
        ),
        "ir_regret_within_preregistered_tolerance": (
            regret_error <= float(tolerance["ir_regret_absolute_error"])
        ),
        "paired_improvement_ci_excludes_zero": (
            aggregate["paired_caama_minus_baseline"]["ci95_low"]
            > float(tolerance["paired_improvement_ci95_low_minimum"])
        ),
        "rival_reversal_increases_ir_regret": (
            shuffled_effect["ci95_low"] > 0
        ),
    }
    return {
        "paper_targets": targets,
        "relative_errors": relative_errors,
        "ir_regret_absolute_error": regret_error,
        "negative_control_paired_regret_effect": shuffled_effect,
        "checks": checks,
        "all_verification_checks_pass": all(checks.values()),
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
rows = list(csv.DictReader((here / "raw_test_samples.csv").open()))
by_seed = defaultdict(list)
for row in rows:
    by_seed[int(row["seed"])].append(row)
assert sorted(by_seed) == [1, 2, 3, 4, 5]
assert all(len(seed_rows) == 20000 for seed_rows in by_seed.values())

keys = [
    "baseline_revenue",
    "caama_revenue",
    "caama_ir_regret",
    "caama_ex_post_ir_revenue",
]
seed_means = {
    key: [
        statistics.fmean(float(row[key]) for row in by_seed[seed])
        for seed in sorted(by_seed)
    ]
    for key in keys
}
paired = [
    seed_means["caama_revenue"][i] - seed_means["baseline_revenue"][i]
    for i in range(5)
]
t4 = 2.7764451051977987
paired_mean = statistics.fmean(paired)
paired_se = statistics.stdev(paired) / math.sqrt(5)
result = {
    "raw_rows": len(rows),
    "seed_counts": {str(k): len(v) for k, v in sorted(by_seed.items())},
    "seed_means": seed_means,
    "paired_caama_minus_baseline": paired,
    "paired_mean": paired_mean,
    "paired_ci95_low": paired_mean - t4 * paired_se,
    "paired_ci95_high": paired_mean + t4 * paired_se,
}
print(json.dumps(result, sort_keys=True))
"""


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
criteria = json.loads((here / "verification_criteria.json").read_text())
independent = json.loads((here / "independent_checker_output.json").read_text())
checks = dict(criteria["checks"])
checks["independent_raw_row_count"] = independent["raw_rows"] == 100000
checks["independent_seed_counts"] = all(
    count == 20000 for count in independent["seed_counts"].values()
)
ok = all(checks.values())
verdict = "VERIFIED" if ok else "BLOCKED"
print(json.dumps({"claim": 5, "verdict": verdict, "ok": ok, "checks": checks}, sort_keys=True))
sys.exit(0 if ok else 1)
"""


def _write_full_claim_bundle(
    route: Path,
    config: dict[str, Any],
    summaries: dict[str, Any],
    aggregate: dict[str, Any],
    controls: dict[str, Any],
    criteria: dict[str, Any],
    started: float,
) -> str:
    direct_parameterization = (
        config.get("parameterization") == "direct_mechanism_space"
    )
    contract = {
        "claim": 5,
        "paper_result": {
            "setting": "Linear Mixture alpha=0.6, 2 bidders x 5 items, asymmetric",
            "randomized_ama_revenue": 1.7135,
            "caama_revenue": 1.9359,
            "caama_ir_regret": 0.0052,
            "caama_ex_post_ir_revenue": 1.8553,
        },
        "paper_source": PAPER_URL,
        "paper_sha256": PAPER_SHA256,
        "anchors": ["S4.T1", "S5.p3", "S5.SS1.p3"],
        "machine_checkable_contract": (
            "Five-seed mean revenues and ex-post-IR revenue must be within 5% "
            "relative error of Table 1, IR regret within 0.003 absolute error, "
            "and the paired CA-AMA improvement 95% t interval must exclude zero."
        ),
        "pre_registered_tolerances": config["verification_tolerances"],
        "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
    }
    _write_json(route / "claim_contract.json", contract)
    _write_text(
        route / "source_audit.md",
        f"""# Claim 5 source audit

- Source: `{PAPER_URL}`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `{PAPER_SHA256}`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1.p3`
- Implementation anchor: `S5.p3`

Table 1 reports Randomized AMA `1.7135`, CA-AMA `1.9359`, parenthetical
IR regret `0.0052`, and ex-post-IR CA-AMA `1.8553` for the asymmetric
two-bidder, five-item, alpha=0.6 setting. The prose calls regret values
"near 0.001"; the exact row is `0.0052`, so this contract uses the row value.

The paper specifies a Bernoulli mixture. Released `generate_data_22` instead
uses a convex interpolation. This route follows the literal paper statement;
the released-code distribution is retained as a distinct alternative route.
""",
    )
    parameterization_method = (
        """- Allocation logits, positive bidder weights, and boosts are optimized
  directly in the 256-menu AMA mechanism space. A per-item softmax over the two
  bidders plus non-allocation enforces exactly the same allocation feasibility
  constraints as AMenuNet. This removes the paper's optimizer parameterization
  but does not change the auction, menu size, scores, or payments."""
        if direct_parameterization
        else """- AMenuNet Transformer architecture, allocations, weights, boosts,
  soft choice, and weighted pivot payments match the released implementation.
- Because contexts are constant identifiers, the dropout-free Transformer is
  evaluated once per update rather than redundantly for each identical context."""
    )
    _write_text(
        route / "method.md",
        f"""# Claim 5 method

- Five deterministic training seeds: 1--5.
- Literal Bernoulli(alpha=0.6) asymmetric mixture from Section 5.1.
- Randomized AMA: 32,000 minibatch updates.
- CA-AMA: 16,000 mutual plus 16,000 pCor-only post-training updates.
- Batch size 2,048; fixed 20,000-sample test set; menu size 256.
{parameterization_method}
- The correlation payment is the paper-stated three-linear-layer ReLU MLP.
- Hard argmax allocation and pivot payments are used for final evaluation.
- Negative pCor outputs are truncated to zero, matching the released evaluator.
- Uncertainty is computed over five independent training-seed means with a
  two-sided 95% Student-t interval.
""",
    )
    _write_json(route / "per_seed_summary.json", summaries)
    _write_json(route / "aggregate_summary.json", aggregate)
    _write_json(route / "negative_control_output.json", controls)
    _write_json(route / "verification_criteria.json", criteria)
    _write_json(route / "config.json", config)
    _write_json(
        route / "exact_command_environment.json",
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
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _write_text(route / "independent_checker.py", _independent_checker_source())
    independent = subprocess.run(
        [sys.executable, str(route / "independent_checker.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    independent_output = json.loads(independent.stdout)
    _write_json(route / "independent_checker_output.json", independent_output)
    _write_text(route / "claim_verifier.py", _claim_verifier_source())
    verifier = subprocess.run(
        [sys.executable, str(route / "claim_verifier.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    verifier_output = {
        "returncode": verifier.returncode,
        "stdout": verifier.stdout.strip(),
        "stderr": verifier.stderr.strip(),
    }
    _write_json(route / "verifier_output.json", verifier_output)
    verdict = "VERIFIED" if verifier.returncode == 0 else "BLOCKED"
    parameterization_limitation = (
        """- This route directly optimizes allocation logits, bidder weights, and
  boosts. It spans the same finite-menu AMA mechanism class but does not use
  AMenuNet's Transformer over-parameterization, so it verifies attainability
  rather than reproducing that optimizer's trajectory."""
        if direct_parameterization
        else """- Constant-context vectorization is algebraically equivalent and is
  regression tested against the released soft-payment implementation, but it is
  still a CPU execution optimization absent from the authors' scripts."""
    )
    _write_text(
        route / "limitations_and_deviations.md",
        f"""# Limitations and deviations

- "32,000 iterations" is interpreted in the standard minibatch-update sense.
  Released scripts instead default to 2,000 outer loops that each consume
  32,768 new samples, so the public code and paper do not define one identical
  counter.
- This route follows the paper's Bernoulli mixture rather than the conflicting
  released convex-interpolation generator.
- Gamma is initialized at 5, one of the paper's stated choices; released data-22
  scripts use 6.
- The paper states a three-layer ReLU pCor MLP, whereas released mutual training
  uses a max-minus-max network and released post-training uses the ReLU MLP.
{parameterization_limitation}
- A non-matching optimization outcome is marked BLOCKED, not FALSIFIED, because
  non-convex training cannot by itself disprove existence of the reported run.
""",
    )
    _write_text(
        route / "EVAL.md",
        f"""# Claim 5 evaluation

- Verdict: **{verdict}**
- Randomized AMA observed mean: `{aggregate['baseline_revenue']['mean']:.6f}`
  (paper `1.7135`)
- CA-AMA observed mean: `{aggregate['caama_revenue']['mean']:.6f}`
  (paper `1.9359`)
- IR regret observed mean: `{aggregate['caama_ir_regret']['mean']:.6f}`
  (paper row `0.0052`)
- Ex-post-IR observed mean: `{aggregate['caama_ex_post_ir_revenue']['mean']:.6f}`
  (paper `1.8553`)
- Paired improvement 95% CI:
  `[{aggregate['paired_caama_minus_baseline']['ci95_low']:.6f}, `
  `{aggregate['paired_caama_minus_baseline']['ci95_high']:.6f}]`
- Verifier exit: `{verifier.returncode}`
""",
    )
    return verdict


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["mode"] == "falsification_audit":
        from claim5_falsification import main as run_falsification_audit

        run_falsification_audit()
        return
    if config["mode"] == "claim4_conditional_support":
        from claim4_conditional import main as run_claim4_conditional

        run_claim4_conditional()
        return
    if config["mode"] == "claim4_cross_item_pcor_pilot":
        from claim4_pcor_pilot import main as run_claim4_pcor_pilot

        run_claim4_pcor_pilot()
        return
    if config["mode"] == "claim4_cross_item_pcor_multiseed":
        from claim4_pcor_pilot import main as run_claim4_pcor_multiseed

        run_claim4_pcor_multiseed()
        return
    if config["mode"] == "claim4_exact_amenunet_pilot":
        from claim4_exact_amenunet import main as run_claim4_exact_amenunet

        run_claim4_exact_amenunet()
        return
    if config["mode"] == "claim4_exact_amenunet_full_seed":
        from claim4_exact_amenunet_full import main as run_claim4_exact_full_seed

        run_claim4_exact_full_seed()
        return
    if config["mode"] == "theory_scope_audit":
        print("EMPIRICAL_TRAIN_STATUS=SKIPPED_THEORY_SCOPE_AUDIT")
        return
    if config["mode"] == "cumulative_release_candidate":
        from campaign_summary import main as run_campaign_summary
        from claim4_conditional import main as run_claim4_conditional
        from claim4_pcor_pilot import main as run_claim4_pcor_multiseed
        from claim5_falsification import main as run_claim5_falsification

        run_claim4_conditional()
        run_claim4_pcor_multiseed()
        run_claim5_falsification()
        run_campaign_summary()
        return
    if config["mode"] not in {"pilot_no_verdict", "full_claim_evidence"}:
        raise ValueError(f"unsupported empirical mode: {config['mode']}")
    started = time.perf_counter()
    claim = int(config["claim"])
    route_name = (
        (
            "route_3_direct_pilot"
            if config.get("parameterization") == "direct_mechanism_space"
            else "route_2_vectorized_pilot"
        )
        if config["mode"] == "pilot_no_verdict"
        else (
            "route_3_direct_mechanism_space"
            if config.get("parameterization") == "direct_mechanism_space"
            else "route_2_full_paper_semantics"
        )
    )
    route = ARTIFACT_ROOT / f"claim_{claim}" / route_name
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
            f"EMPIRICAL_RESULT seed={seed} "
            f"baseline={summary['baseline_revenue']['mean']:.6f} "
            f"caama={summary['caama_revenue']['mean']:.6f} "
            f"ir_regret={summary['caama_ir_regret']['mean']:.6f} "
            f"ex_post_ir={summary['caama_ex_post_ir_revenue']['mean']:.6f}"
        )
    _write_csv(route / "raw_test_samples.csv", all_raw)
    _write_csv(route / "learning_curves.csv", all_curves)
    if config["mode"] == "pilot_no_verdict":
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
            """# Direct mechanism-space pilot

This one-seed, 1,000-total-update pilot directly optimizes feasible allocation
logits, positive weights, and boosts in the same 256-menu AMA mechanism space.
It retains the literal paper distribution, full batch and test sizes, pivot
payments, pCor MLP, and IR accounting. It is a throughput and optimization
diagnostic, not Table-1 claim evidence.
""",
        )
        _write_text(
            route / "STATUS.md",
            """# Status

- Status: **PILOT ONLY — NO CLAIM VERDICT**
- Full 32,000-update, five-seed evidence is reserved for a child after this
  implementation and throughput check passes.
""",
        )
        verdict = "PILOT_ONLY_NO_VERDICT"
    else:
        aggregate = aggregate_seed_results(summaries)
        criteria = verification_criteria(
            config, aggregate, controls
        )
        verdict = _write_full_claim_bundle(
            route,
            config,
            summaries,
            aggregate,
            controls,
            criteria,
            started,
        )
        print(
            "CLAIM_5_AGGREGATE "
            f"baseline={aggregate['baseline_revenue']['mean']:.6f} "
            f"caama={aggregate['caama_revenue']['mean']:.6f} "
            f"regret={aggregate['caama_ir_regret']['mean']:.6f} "
            f"ex_post_ir={aggregate['caama_ex_post_ir_revenue']['mean']:.6f}"
        )
    print(f"EMPIRICAL_TRAIN_MODE={config['mode']}")
    print(f"CLAIM_5_VERDICT={verdict}")
    print(f"EMPIRICAL_TRAIN_RUNTIME_SECONDS={time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
