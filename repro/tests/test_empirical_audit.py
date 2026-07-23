#!/usr/bin/env python3
"""Tests for literal Table-1 data distributions and discrepancy controls."""

import numpy as np

import empirical_audit as ea


def test_dirichlet_contract_shape_and_total():
    values = ea.dirichlet_value_share(4096, 3, 10, 0.5, 17)
    assert values.shape == (4096, 3, 10)
    totals = values.sum(axis=1)
    assert np.all(totals >= 0.5)
    assert np.all(totals <= 1.0)


def test_literal_linear_mixture_has_probability_mass_on_relation():
    values, dependent = ea.linear_mixture_asymmetric(50_000, 5, 0.6, 19)
    residual = 4.0 * values[:, 1] + values[:, 0] - 1.0
    observed = np.mean(np.abs(residual) < 1e-12)
    assert abs(observed - 0.6) < 0.01
    assert abs(dependent.mean() - 0.6) < 0.01


def test_official_convex_generator_is_not_paper_bernoulli_mixture():
    paper, _ = ea.linear_mixture_asymmetric(50_000, 5, 0.6, 23)
    convex = ea.upstream_convex_asymmetric(50_000, 5, 0.6, 23)
    paper_rate = np.mean(
        np.abs(4.0 * paper[:, 1] + paper[:, 0] - 1.0) < 1e-12
    )
    convex_rate = np.mean(
        np.abs(4.0 * convex[:, 1] + convex[:, 0] - 1.0) < 1e-12
    )
    assert paper_rate > 0.59
    assert convex_rate == 0.0
