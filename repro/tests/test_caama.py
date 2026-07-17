#!/usr/bin/env python3
"""Exact-identity tests for CA-AMA (TA3NDHgNJh, Theorem 3.3)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import caama as cm


def test_REV_F_closed_equals_numeric():
    """Optimal revenue E[v1] = eps ln(1/eps)/(1-eps): closed form == numerical integration."""
    for e in [0.1, 0.01, 0.001]:
        assert abs(cm.REV_F_closed(e) - cm.REV_F_numeric(e)) < 1e-9


def test_v1_dominates_v2():
    """In the constructed instance v1 >= v2 always (bidder 1 always wins the efficient allocation)."""
    for e in [0.1, 0.05, 0.01]:
        for v1 in np.linspace(e, 1.0, 50):
            assert v1 >= cm.v2_of(v1, e) - 1e-12


def test_c1_classic_ama_ratio_decreases_to_zero():
    """Classic-AMA revenue / REV_F -> 0 as eps -> 0."""
    eps_list = [0.1, 0.03, 0.01, 0.003, 0.001]
    ratios = [cm.classic_ama_revenue(e) / cm.REV_F_closed(e) for e in eps_list]
    assert all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1))
    assert ratios[-1] < 0.2


def test_c1_best_classic_ama_still_poor():
    """Even the best classic AMA (optimized reserve) is arbitrarily poor vs REV_F."""
    for e in [0.01, 0.001]:
        assert cm.best_classic_ama_with_reserve(e) / cm.REV_F_closed(e) < 0.3


def test_c2_caama_equals_optimal():
    """CA-AMA (full-surplus extraction) revenue == REV_F exactly."""
    for e in [0.1, 0.01, 0.001]:
        assert abs(cm.caama_revenue(e) - cm.REV_F_closed(e)) < 1e-9


def test_iid_full_surplus_not_extractable():
    """Negative control: without correlation, full surplus > Myerson optimal (not extractable)."""
    from scipy import integrate
    full = integrate.dblquad(lambda v2, v1: max(v1, v2), 0, 1, 0, 1)[0]   # E[max]=2/3
    assert 5.0 / 12.0 < full   # Myerson (0.4167) < full surplus (0.6667)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
