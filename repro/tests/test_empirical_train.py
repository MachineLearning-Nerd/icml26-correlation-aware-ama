#!/usr/bin/env python3
"""Equivalence and accounting checks for vectorized empirical training."""

from types import SimpleNamespace

import torch

import empirical_train as et


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
