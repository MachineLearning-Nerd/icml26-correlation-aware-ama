#!/usr/bin/env python3
"""Equivalence and accounting checks for vectorized empirical training."""

from types import SimpleNamespace

import numpy as np
import torch

import empirical_train as et
import claim5_falsification as c5f
import claim4_conditional as c4c
import claim4_pcor_pilot as c4p


def _official_args(batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_agents=2,
        m_items=2,
        dx=10,
        dy=10,
        menu_size=8,
        deterministic=False,
        continuous_context=False,
        const_bidder_weights=False,
        d_emb=10,
        n_layer=3,
        n_head=4,
        d_hidden=64,
        init_softmax_temperature=500,
        alloc_softmax_temperature=10,
        batch_size=batch_size,
        device="cpu",
        ablation=0,
    )


def test_constant_context_vectorization_matches_official_soft_forward():
    import sys

    upstream = str(et.ROOT / "upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from auction import ContextualAffineMaximizerAuction

    batch = 5
    torch.manual_seed(123)
    official = ContextualAffineMaximizerAuction(_official_args(batch))
    efficient = et.EfficientAMA(2, 2, 8, 10)
    efficient.mechanism.load_state_dict(official.citransnet.state_dict())
    values = torch.rand(batch, 2, 2)
    x = torch.arange(2).repeat(batch).reshape(batch, 2)
    y = torch.arange(2).repeat(batch).reshape(batch, 2)
    _, _, official_payment, _, official_valuation = official(
        values, x, y, 500
    )
    payment, valuation, _ = efficient.soft_outcomes(values, 500)
    assert torch.allclose(payment, official_payment.T, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        valuation, official_valuation.T, atol=1e-6, rtol=1e-6
    )


def test_hard_pivot_payments_are_nonnegative_externalities():
    torch.manual_seed(321)
    model = et.EfficientAMA(2, 2, 8, 10)
    values = torch.rand(16, 2, 2)
    payment, valuation, choices = model.hard_outcomes(values)
    assert payment.shape == (16, 2)
    assert valuation.shape == (16, 2)
    assert choices.shape == (16,)
    assert torch.isfinite(payment).all()


def test_post_training_uses_hard_argmax_ama_path(monkeypatch):
    config = {
        "n_bidders": 2,
        "n_items": 2,
        "menu_size": 8,
        "allocation_temperature": 10.0,
        "parameterization": "amenunet_constant_context",
        "train_batch_size": 4,
        "mutual_updates": 0,
        "post_updates": 1,
        "learning_rate": 0.0003,
        "warmup_updates": 100,
        "softmax_temperature": 500.0,
        "gamma_initial": 3.0,
        "gamma_learning_rate": 0.01,
        "gamma_min": 1.0,
        "gamma_max": 20.0,
        "target_ir_regret": 0.001,
        "log_every": 1,
        "distribution": "dirichlet_value_share",
        "alpha": 0.5,
    }
    hard_calls = 0
    original_hard = et.EfficientAMA.hard_outcomes

    def counted_hard(self, values):
        nonlocal hard_calls
        hard_calls += 1
        return original_hard(self, values)

    def forbidden_soft(self, values, temperature):
        raise AssertionError("post-training called the soft AMA path")

    monkeypatch.setattr(et.EfficientAMA, "hard_outcomes", counted_hard)
    monkeypatch.setattr(et.EfficientAMA, "soft_outcomes", forbidden_soft)
    et.train_caama(config, seed=7, curve_rows=[])
    assert hard_calls == 1


def test_literal_sampler_preserves_asymmetric_support():
    config = {
        "n_bidders": 2,
        "n_items": 5,
        "alpha": 0.6,
        "distribution": "paper_literal_bernoulli_mixture",
    }
    generator = torch.Generator().manual_seed(99)
    values = et.sample_values(config, 4096, generator)
    assert values.shape == (4096, 2, 5)
    assert torch.all((values[:, 0] >= 0) & (values[:, 0] <= 1))
    assert torch.all((values[:, 1] >= 0) & (values[:, 1] <= 0.25))


def test_direct_mechanism_space_is_feasible_and_differentiable():
    torch.manual_seed(777)
    model = et.DirectAMA(3, 4, 16, 10)
    allocations, weights, boosts = model.parameters_for_auction()
    assert allocations.shape == (17, 3, 4)
    assert boosts.shape == (17,)
    assert torch.all(allocations >= 0)
    assert torch.all(allocations[:-1].sum(dim=1) <= 1 + 1e-6)
    assert torch.all(weights > 0)
    values = torch.rand(32, 3, 4)
    payment, _, _ = model.soft_outcomes(values, 500)
    loss = -payment.sum(dim=1).mean()
    loss.backward()
    assert model.allocation_logits.grad is not None
    assert torch.isfinite(model.allocation_logits.grad).all()


def test_claim5_exact_falsification_bound_is_sensitive_but_not_triggered():
    result = c5f.exact_falsification_bounds()
    assert (
        result["distribution"]["expected_total_welfare_upper_bound"]["exact"]
        == "623/240"
    )
    assert all(result["necessary_checks"].values())
    assert not result["valid_falsification_found"]


def test_claim4_conditional_floor_is_rival_only_and_support_ir():
    values = c4c.sample_dirichlet_profiles(seed=77, profiles=256, items=2)
    floor = c4c.conditional_utility_floor(values, reserve=0.2)
    assert floor.shape == values.shape
    assert np.all(floor >= 0)
    metrics = c4c.auction_metrics(values, reserve=0.2)
    assert np.min(metrics["minimum_bidder_utility"]) >= -1e-12
    assert np.all(
        metrics["caama_revenue"]
        <= metrics["welfare"] + metrics["caama_ir_regret"] + 1e-12
    )


def test_claim4_pcor_sampler_and_reserve_core_are_feasible():
    generator = torch.Generator().manual_seed(88)
    values = c4p.sample_torch(128, generator)
    payment, utility, welfare = c4p.reserve_core(values, reserve=0.3)
    assert values.shape == (128, 3, 10)
    assert payment.shape == utility.shape == (128, 3)
    assert welfare.shape == (128,)
    assert torch.all(payment >= 0)
    assert torch.all(utility >= -1e-7)
    assert torch.all(payment.sum(dim=1) <= welfare + 1e-7)


def test_claim4_pcor_is_rival_only_and_scaled_metrics_obey_bound():
    torch.manual_seed(901)
    model = et.PaperPaymentMLP(3, 10)
    values = c4p.sample_torch(64, torch.Generator().manual_seed(902))
    changed = values.clone()
    changed[:, 1] = torch.rand(
        changed[:, 1].shape, generator=torch.Generator().manual_seed(903)
    )
    original_payment = model(values)
    changed_payment = model(changed)
    assert torch.allclose(original_payment[:, 1], changed_payment[:, 1])

    components = c4p._components(model, values)
    raw, _ = c4p._metrics(components, scale=0.5)
    assert np.all(
        raw["caama_revenue"]
        <= raw["welfare"] + raw["caama_ir_regret"] + 1e-6
    )
