#!/usr/bin/env python3
"""Unit tests for the quantified Claims 1--3 evidence harness."""
from decimal import Decimal

import theory_campaign as tc


def test_claim_1_explicit_all_n_certificate():
    for delta in ["0.5", "0.1", "0.01", "0.001"]:
        for n in [2, 3, 10, 50]:
            row = tc.construction_certificate(delta, n)
            assert row["assumptions_hold"]
            assert row["formula_agrees"]
            assert Decimal(str(row["upper_bound_ratio"])) < Decimal(delta)


def test_claim_1_rejects_out_of_domain_n_1():
    try:
        tc.construction_certificate("0.1", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("n=1 must not be silently counted")


def test_claim_1_negative_control_fails_contract():
    row = tc.bad_construction_control("0.1")
    assert row["rejected_as_intended"]
    assert not row["contract_holds"]


def test_claim_2_correlated_payment_extracts_pointwise():
    for n in [2, 3, 5, 10]:
        assert tc.correlated_full_extraction_case("0.25", n)[
            "full_extraction_holds"
        ]


def test_claim_2_independent_transform_dominates():
    for seed, n, positive in [
        (1103, 2, True),
        (2207, 3, False),
        (3301, 4, True),
    ]:
        row = tc.independent_transform_case(seed, n, positive)
        assert row["pointwise_transform_dominates"]


def test_claim_3_rival_only_shift_preserves_dsic():
    row = tc.dsic_property_case(1103, 3, 2, 12)
    assert row["dsic_holds"]
    assert row["max_gain_shift_error"] <= 1e-12
