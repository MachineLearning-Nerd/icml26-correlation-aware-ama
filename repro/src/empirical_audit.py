#!/usr/bin/env python3
"""Source-faithful data checks and CPU profiling for Table-1 Claims 4--5."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".openresearch" / "artifacts"
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2602.09455"
PAPER_RETRIEVED = "2026-07-23T15:56:49Z"
PAPER_SHA256 = "2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f"
UPSTREAM_SHA = "ed2af19ed02c70b58efdf705635981241222d045"
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)
SEEDS = [1, 2, 3, 4, 5]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


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


def dirichlet_value_share(
    sample_num: int,
    n_bidders: int,
    n_items: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Literal Section 5.1 generator: T_j~U[.5,1], w_j~Dir(alpha)."""
    rng = np.random.default_rng(seed)
    totals = rng.uniform(0.5, 1.0, size=(sample_num, n_items))
    shares = rng.dirichlet(
        np.full(n_bidders, alpha), size=(sample_num, n_items)
    )
    return (shares * totals[..., None]).transpose(0, 2, 1)


def linear_mixture_asymmetric(
    sample_num: int,
    n_items: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal Bernoulli-mixture distribution in Section 5.1."""
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(0.0, 1.0, size=(sample_num, n_items))
    dependent = rng.uniform(size=(sample_num, n_items)) < alpha
    independent = rng.uniform(0.0, 0.25, size=(sample_num, n_items))
    v2 = np.where(dependent, (1.0 - v1) / 4.0, independent)
    return np.stack((v1, v2), axis=1), dependent


def upstream_convex_asymmetric(
    sample_num: int,
    n_items: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Reimplementation of official generate_data_22 for a negative control."""
    rng = np.random.default_rng(seed)
    v1_base = rng.uniform(0.0, 1.0, size=(sample_num, n_items))
    v2_base = 1.0 - v1_base
    special = rng.uniform(0.0, 1.0, size=(sample_num, 2, n_items))
    v1 = alpha * v1_base + (1.0 - alpha) * special[:, 0]
    v2 = (alpha * v2_base + (1.0 - alpha) * special[:, 1]) / 4.0
    return np.stack((v1, v2), axis=1)


def data_contract_checks(sample_num: int = 200_000) -> dict[str, Any]:
    alpha_dir = 0.5
    n_bidders, n_items = 3, 10
    dir_values = dirichlet_value_share(
        sample_num, n_bidders, n_items, alpha_dir, seed=20260723
    )
    per_item_total = dir_values.sum(axis=1)
    flat_dir = dir_values.transpose(0, 2, 1).reshape(-1, n_bidders)
    empirical_cov = float(np.cov(flat_dir[:, 0], flat_dir[:, 1], ddof=1)[0, 1])
    expected_t2 = (0.5**2 + 0.5 * 1.0 + 1.0**2) / 3.0
    expected_cross_share = alpha_dir / (
        n_bidders * (n_bidders * alpha_dir + 1.0)
    )
    expected_cov = expected_t2 * expected_cross_share - (0.75 / n_bidders) ** 2

    alpha_mix = 0.6
    mix_values, dependent = linear_mixture_asymmetric(
        sample_num, 5, alpha_mix, seed=20260724
    )
    flat_v1 = mix_values[:, 0].reshape(-1)
    flat_v2 = mix_values[:, 1].reshape(-1)
    paper_cov = float(np.cov(flat_v1, flat_v2, ddof=1)[0, 1])
    expected_mix_cov = -alpha_mix / 48.0
    residual = 4.0 * mix_values[:, 1] + mix_values[:, 0] - 1.0
    exact_relation_rate = float(np.mean(np.abs(residual) < 1e-12))

    convex_values = upstream_convex_asymmetric(
        sample_num, 5, alpha_mix, seed=20260724
    )
    convex_residual = (
        4.0 * convex_values[:, 1] + convex_values[:, 0] - 1.0
    )
    convex_relation_rate = float(np.mean(np.abs(convex_residual) < 1e-12))
    expected_paper_v2_variance = 1.0 / 192.0
    empirical_paper_v2_variance = float(np.var(flat_v2))
    empirical_convex_v2_variance = float(np.var(convex_values[:, 1]))

    checks = {
        "dirichlet": {
            "shape": list(dir_values.shape),
            "paper_shape": [sample_num, 3, 10],
            "minimum_realized_total_value": float(per_item_total.min()),
            "maximum_realized_total_value": float(per_item_total.max()),
            "mean_total_value": float(per_item_total.mean()),
            "expected_mean_total_value": 0.75,
            "mean_bidder_item_value": float(dir_values.mean()),
            "expected_mean_bidder_item_value": 0.25,
            "empirical_cross_bidder_covariance": empirical_cov,
            "analytic_cross_bidder_covariance": expected_cov,
            "contract_holds": bool(
                dir_values.shape == (sample_num, 3, 10)
                and float(per_item_total.min()) >= 0.5 - 1e-12
                and float(per_item_total.max()) <= 1.0 + 1e-12
                and abs(float(per_item_total.mean()) - 0.75) < 0.002
                and abs(float(dir_values.mean()) - 0.25) < 0.001
                and abs(empirical_cov - expected_cov) < 0.001
            ),
        },
        "linear_mixture_asymmetric": {
            "shape": list(mix_values.shape),
            "paper_shape": [sample_num, 2, 5],
            "empirical_dependent_fraction": float(dependent.mean()),
            "expected_dependent_fraction": alpha_mix,
            "empirical_v1_mean": float(flat_v1.mean()),
            "expected_v1_mean": 0.5,
            "empirical_v2_mean": float(flat_v2.mean()),
            "expected_v2_mean": 0.125,
            "empirical_covariance": paper_cov,
            "analytic_covariance": expected_mix_cov,
            "empirical_v2_variance": empirical_paper_v2_variance,
            "analytic_v2_variance": expected_paper_v2_variance,
            "exact_linear_relation_rate": exact_relation_rate,
            "contract_holds": bool(
                mix_values.shape == (sample_num, 2, 5)
                and abs(float(dependent.mean()) - alpha_mix) < 0.003
                and abs(paper_cov - expected_mix_cov) < 0.0004
                and abs(empirical_paper_v2_variance - expected_paper_v2_variance)
                < 0.0001
                and abs(exact_relation_rate - alpha_mix) < 0.003
            ),
        },
        "official_code_negative_control": {
            "official_generator": "upstream/gen_values.py::generate_data_22",
            "official_formula": (
                "v=alpha*base+(1-alpha)*special; then bidder 2 divided by 4"
            ),
            "paper_formula": (
                "Bernoulli(alpha): v2=(1-v1)/4; otherwise v2~U[0,1/4]"
            ),
            "convex_exact_linear_relation_rate": convex_relation_rate,
            "convex_v2_variance": empirical_convex_v2_variance,
            "paper_v2_variance": expected_paper_v2_variance,
            "distinguishes_paper_distribution": bool(
                exact_relation_rate > 0.59
                and convex_relation_rate < 1e-5
                and abs(empirical_convex_v2_variance - expected_paper_v2_variance)
                > 0.001
            ),
        },
    }
    checks["all_contracts_hold"] = bool(
        checks["dirichlet"]["contract_holds"]
        and checks["linear_mixture_asymmetric"]["contract_holds"]
        and checks["official_code_negative_control"][
            "distinguishes_paper_distribution"
        ]
    )
    return checks


def _profile_args(n: int, m: int, menu: int, batch: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_agents=n,
        m_items=m,
        dx=10,
        dy=10,
        menu_size=menu,
        deterministic=False,
        continuous_context=False,
        const_bidder_weights=False,
        d_emb=10,
        n_layer=3,
        n_head=4,
        d_hidden=64,
        init_softmax_temperature=500,
        alloc_softmax_temperature=10,
        batch_size=batch,
        device="cpu",
        ablation=0,
    )


def profile_training_batch(
    *,
    name: str,
    n: int,
    m: int,
    menu: int,
    batch: int,
    seed: int,
) -> dict[str, Any]:
    """One faithful joint CA-AMA forward/backward on CPU for capacity planning."""
    upstream_path = str(ROOT / "upstream")
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    from auction import ContextualAffineMaximizerAuction
    from net import Payment_Cor_max_min

    torch.manual_seed(seed)
    args = _profile_args(n, m, menu, batch)
    started = time.perf_counter()
    model = ContextualAffineMaximizerAuction(args)
    payment_model = Payment_Cor_max_min(args)
    init_seconds = time.perf_counter() - started
    if name == "claim_4_dirichlet":
        values_np = dirichlet_value_share(batch, n, m, 0.5, seed)
    else:
        values_np, _ = linear_mixture_asymmetric(batch, m, 0.6, seed)
    values = torch.tensor(values_np, dtype=torch.float32)
    x = torch.arange(n).repeat(batch).reshape(batch, n).long()
    y = torch.arange(m).repeat(batch).reshape(batch, m).long()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(payment_model.parameters()), lr=3e-4
    )
    optimizer.zero_grad()
    started = time.perf_counter()
    _, _, payment, _, valuation = model(values, x, y, 500)
    p_cor = payment_model(values, x, y)
    utility = valuation - payment - p_cor
    loss = -(payment + p_cor).sum(0).mean() + 3.0 * torch.clamp(
        -utility, min=0
    ).sum(0).mean()
    loss.backward()
    optimizer.step()
    batch_seconds = time.perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024.0 if platform.system() != "Darwin" else 1024.0 * 1024.0
    )
    return {
        "name": name,
        "n_bidders": n,
        "n_items": m,
        "menu_size": menu,
        "profile_batch_size": batch,
        "init_seconds": init_seconds,
        "joint_forward_backward_step_seconds": batch_seconds,
        "seconds_per_sample": batch_seconds / batch,
        "loss": float(loss.detach()),
        "peak_process_rss_mb": peak_rss_mb,
        "paper_batch_size": 1024 if name == "claim_4_dirichlet" else 2048,
        "paper_total_iterations": 32_000,
        "profile_only_not_claim_evidence": True,
    }


def cpu_profiles() -> dict[str, Any]:
    profiles = [
        profile_training_batch(
            name="claim_4_dirichlet", n=3, m=10, menu=2048, batch=4, seed=1
        ),
        profile_training_batch(
            name="claim_5_linear_asymmetric",
            n=2,
            m=5,
            menu=256,
            batch=32,
            seed=1,
        ),
    ]
    for row in profiles:
        row["linear_seconds_estimate_for_32000_minibatches"] = (
            row["seconds_per_sample"] * row["paper_batch_size"] * 32_000
        )
        row["linear_hours_estimate_for_32000_minibatches"] = (
            row["linear_seconds_estimate_for_32000_minibatches"] / 3600.0
        )
        row["estimate_warning"] = (
            "Small-batch linear extrapolation only; full-batch kernels, memory, "
            "evaluation, five seeds, baseline training, and post-training differ."
        )
    return {
        "profiles": profiles,
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "memory_bytes": psutil.virtual_memory().total,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "source_ambiguities": [
            (
                "Paper says 32,000 iterations and total mutual+post-training, but "
                "released scripts default to 2,000 outer steps and each outer step "
                "processes train_sample_num=32,768 through many minibatches."
            ),
            (
                "Paper does not state the 3x10 menu size; released scripts omit "
                "3 bidders x 10 items. The profile uses the scripts' apparent 2048 cap."
            ),
            (
                "Paper says a three-layer ReLU pCor MLP; released mutual-training "
                "code uses Payment_Cor_max_min, while post-training uses Payment_Cor."
            ),
        ],
    }


def _source_audit(claim: int) -> str:
    if claim == 4:
        exact = (
            "Table 1: Dirichlet Value Share alpha=0.5, 3x10, Randomized AMA "
            "3.1363, CA-AMA 3.6205 (IR regret 0.0031), ex-post-IR CA-AMA 3.5623."
        )
        contract = (
            "Using 3 bidders, 10 items, T_j~U[0.5,1], and "
            "w_j~Dirichlet(0.5,0.5,0.5), reproduce five-seed test revenue and "
            "uncertainty for equally sized Randomized AMA and CA-AMA."
        )
    else:
        exact = (
            "Table 1: Linear Mixture alpha=0.6, 2x5 Asym, Randomized AMA "
            "1.7135, CA-AMA 1.9359 (IR regret 0.0052), ex-post-IR CA-AMA 1.8553."
        )
        contract = (
            "Using 2 bidders, 5 items and the literal Bernoulli(alpha=0.6) "
            "asymmetric mixture, reproduce five-seed test revenue, uncertainty, "
            "and empirical IR regret for equally sized mechanisms."
        )
    return f"""# Source audit

- Source: `{PAPER_URL}`
- Retrieved: `{PAPER_RETRIEVED}`
- SHA-256: `{PAPER_SHA256}`
- Anchors: Table 1 `S4.T1`; implementation `S5.p3`; distribution `S5.SS1`
- Exact source statement: {exact}
- Claim contract: {contract}
- Official code revision audited: `{UPSTREAM_SHA}`

This profiling node does not issue a claim verdict. It validates the input
distribution contract and measures feasibility before full training.
"""


def main() -> None:
    started = time.perf_counter()
    checks = data_contract_checks()
    if not checks["all_contracts_hold"]:
        raise AssertionError(f"paper data contract failed: {checks}")
    profiles = cpu_profiles()
    common = {
        "fixed_command": FIXED_COMMAND,
        "git_sha": _git_sha(),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "seeds_reserved_for_full_campaign": SEEDS,
    }
    for claim in (4, 5):
        directory = ARTIFACT_ROOT / f"claim_{claim}" / "route_1_data_and_profile"
        _write_text(directory / "source_audit.md", _source_audit(claim))
        _write_json(directory / "data_contract_checks.json", checks)
        _write_json(directory / "cpu_profile.json", profiles)
        _write_json(directory / "exact_command_environment.json", common)
        _write_text(
            directory / "STATUS.md",
            f"""# Claim {claim} route 1 status

- Status: **PROFILE ONLY — NO VERDICT**
- Paper-faithful data generator: passed analytic moment checks.
- Source/code discrepancies: detected and preserved as alternative routes.
- Full training evidence: not yet run.
""",
        )
    print("\n" + "=" * 74)
    print("TABLE-1 DATA CONTRACTS AND LOCAL CPU PROFILE")
    print("=" * 74)
    print(f"EMPIRICAL_DATA_CONTRACTS_PASS={checks['all_contracts_hold']}")
    print(
        "CLAIM_4_PROFILE_SECONDS="
        f"{profiles['profiles'][0]['joint_forward_backward_step_seconds']:.6f}"
    )
    print(
        "CLAIM_5_PROFILE_SECONDS="
        f"{profiles['profiles'][1]['joint_forward_backward_step_seconds']:.6f}"
    )
    print("CLAIM_4_STATUS=PROFILE_ONLY_NO_VERDICT")
    print("CLAIM_5_STATUS=PROFILE_ONLY_NO_VERDICT")
    print(f"EMPIRICAL_AUDIT_RUNTIME_SECONDS={time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
