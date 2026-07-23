#!/usr/bin/env python3
"""Regression tests for cumulative evidence record parsing."""

import campaign_summary as cs


def test_checks_payload_accepts_direct_and_wrapped_verifiers():
    checks = {"numeric_contract": True, "exact_core_available": False}
    assert cs._checks_payload({"checks": checks}) == checks
    assert cs._checks_payload({
        "returncode": 1,
        "stdout": (
            '{"checks": {"numeric_contract": true, '
            '"exact_core_available": false}}'
        ),
    }) == checks
