#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
raw = json.loads((here / "raw_results.json").read_text())
independent = json.loads((here / "independent_checker_output.json").read_text())
negative = json.loads((here / "negative_control_output.json").read_text())

checks = {
    1: raw.get("all_contracts_hold")
       and raw.get("literal_any_n_claim_falsified")
       and independent.get("all_30_certificates_pass")
       and independent.get("n_1_finite_support_checks_pass")
       and negative.get("weak_parameterization_rejected")
       and negative.get("appendix_construction_rejects_n_1")
       and negative["positive_revenue_n_1_falsification"]["valid_literal_scope_falsification"],
    2: raw.get("correlated_part_holds")
       and raw.get("independent_part_holds")
       and raw.get("n_1_independent_equality_holds")
       and raw.get("literal_any_n_correlated_part_falsified")
       and independent.get("all_cases_pass")
       and independent.get("n_1_finite_support_checks_pass")
       and negative["positive_revenue_n_1_falsification"]["valid_literal_scope_falsification"]
       and not negative["paper_formula_typo_detected"]["printed_formula_matches_v1_when_eta_1_lt_eta"],
    3: raw.get("all_cases_pass")
       and independent.get("all_cases_pass")
       and negative["own_bid_dependent_payment"]["rejected_as_intended"]
       and not negative["own_bid_dependent_payment"]["dsic_holds"],
}
ok = bool(checks[1])
print(json.dumps({"claim": 1, "verdict": "FALSIFIED" if ok else "BLOCKED", "ok": ok}))
sys.exit(0 if ok else 1)
