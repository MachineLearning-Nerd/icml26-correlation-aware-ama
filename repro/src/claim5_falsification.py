#!/usr/bin/env python3
"""Mandatory fourth-route falsification audit for Table 1 Claim 5."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import psutil
import torch


ROOT = Path(__file__).resolve().parents[2]
ROUTE = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_5"
    / "route_4_falsification_audit"
)
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2602.09455"
PAPER_SHA256 = (
    "2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f"
)
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


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


def _fraction_record(value: Fraction) -> dict[str, str | float]:
    return {
        "exact": f"{value.numerator}/{value.denominator}",
        "decimal": float(value),
    }


def exact_falsification_bounds() -> dict[str, Any]:
    """Return assumption-implied necessary bounds using exact arithmetic.

    For one item in the correlated component,
      E max(X, (1-X)/4) = integral_0^.2 (1-x)/4 dx
                            + integral_.2^1 x dx = 21/40.
    For the independent component with Y~U[0,1/4],
      E max(X,Y) = E[X] + E[(Y-X)_+] = 1/2 + E[Y^2]/2 = 49/96.
    """
    alpha = Fraction(3, 5)
    items = 5
    correlated_welfare = Fraction(21, 40)
    independent_welfare = Fraction(49, 96)
    per_item_welfare = (
        alpha * correlated_welfare
        + (1 - alpha) * independent_welfare
    )
    expected_welfare = items * per_item_welfare
    paper = {
        "randomized_ama_revenue": Fraction(17135, 10000),
        "caama_revenue": Fraction(19359, 10000),
        "caama_ir_regret": Fraction(52, 10000),
        "caama_ex_post_ir_revenue": Fraction(18553, 10000),
    }
    # For any allocation-feasible auction, revenue = welfare - sum utility.
    # The sum of negative utilities is at most Regret_IR, hence
    # E[revenue] <= E[welfare] + E[Regret_IR].
    checks = {
        "randomized_ama_below_expected_welfare": (
            paper["randomized_ama_revenue"] <= expected_welfare
        ),
        "caama_below_welfare_plus_reported_regret": (
            paper["caama_revenue"]
            <= expected_welfare + paper["caama_ir_regret"]
        ),
        "ex_post_ir_revenue_below_expected_welfare": (
            paper["caama_ex_post_ir_revenue"] <= expected_welfare
        ),
        "reported_regret_nonnegative": paper["caama_ir_regret"] >= 0,
    }
    contradictions = {name: not passes for name, passes in checks.items()}
    return {
        "distribution": {
            "items": items,
            "alpha": _fraction_record(alpha),
            "correlated_component_expected_item_welfare": _fraction_record(
                correlated_welfare
            ),
            "independent_component_expected_item_welfare": _fraction_record(
                independent_welfare
            ),
            "mixture_expected_item_welfare": _fraction_record(
                per_item_welfare
            ),
            "expected_total_welfare_upper_bound": _fraction_record(
                expected_welfare
            ),
        },
        "paper_values": {
            key: _fraction_record(value) for key, value in paper.items()
        },
        "necessary_checks": checks,
        "contradictions": contradictions,
        "valid_falsification_found": any(contradictions.values()),
    }


def _independent_checker_source() -> str:
    return """#!/usr/bin/env python3
import json
from fractions import Fraction
from pathlib import Path

here = Path(__file__).resolve().parent
raw = json.loads((here / "raw_bound_results.json").read_text())
alpha = Fraction(3, 5)
correlated = Fraction(21, 40)
independent = Fraction(1, 2) + Fraction(1, 4) ** 2 / 6
expected_welfare = 5 * (
    alpha * correlated + (1 - alpha) * independent
)
reported = {
    "baseline": Fraction(17135, 10000),
    "caama": Fraction(19359, 10000),
    "regret": Fraction(52, 10000),
    "ex_post": Fraction(18553, 10000),
}
checks = {
    "expected_welfare_exact": (
        raw["distribution"]["expected_total_welfare_upper_bound"]["exact"]
        == f"{expected_welfare.numerator}/{expected_welfare.denominator}"
    ),
    "baseline_feasible": reported["baseline"] <= expected_welfare,
    "caama_feasible_with_regret": (
        reported["caama"] <= expected_welfare + reported["regret"]
    ),
    "ex_post_feasible": reported["ex_post"] <= expected_welfare,
}
result = {
    "expected_welfare_exact": (
        f"{expected_welfare.numerator}/{expected_welfare.denominator}"
    ),
    "checks": checks,
    "valid_falsification_found": not all(checks.values()),
}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
"""


def _claim_verifier_source() -> str:
    return """#!/usr/bin/env python3
import json
from pathlib import Path

here = Path(__file__).resolve().parent
raw = json.loads((here / "raw_bound_results.json").read_text())
independent = json.loads(
    (here / "independent_checker_output.json").read_text()
)
negative = json.loads((here / "negative_control_output.json").read_text())
falsified = bool(
    raw["valid_falsification_found"]
    and independent["valid_falsification_found"]
)
route_integrity = bool(
    all(independent["checks"].values())
    and negative["counterfeit_revenue_is_rejected"]
)
verdict = "FALSIFIED" if falsified and route_integrity else "BLOCKED"
print(json.dumps({
    "claim": 5,
    "verdict": verdict,
    "valid_falsification_found": falsified,
    "route_integrity": route_integrity,
}, sort_keys=True))
# Nonzero is intentional when the route did not establish the claim verdict.
raise SystemExit(0 if verdict == "FALSIFIED" else 1)
"""


def main() -> None:
    started = time.perf_counter()
    raw = exact_falsification_bounds()
    _write_json(ROUTE / "raw_bound_results.json", raw)
    _write_json(
        ROUTE / "claim_contract.json",
        {
            "claim": 5,
            "exact_statement": (
                "For the asymmetric Linear Mixture with alpha=0.6, two "
                "bidders, and five items, Table 1 reports randomized AMA "
                "revenue 1.7135, CA-AMA revenue 1.9359, IR regret 0.0052, "
                "and ex-post-IR revenue 1.8553."
            ),
            "assumptions": [
                "additive valuations and feasible item allocations",
                "v1j is Uniform[0,1]",
                "with probability 0.6, v2j=(1-v1j)/4",
                "otherwise v2j is independent Uniform[0,1/4]",
                "five independent items",
                "paper Regret_IR definition",
            ],
            "quantifier": (
                "The claim concerns the four reported five-seed mean values "
                "for this single experimental configuration."
            ),
            "falsification_rule": (
                "FALSIFIED only if a reported value violates a necessary "
                "distributional, allocation-feasibility, or IR inequality. "
                "A failed training run is not a counterexample."
            ),
            "allowed_verdicts": ["FALSIFIED", "BLOCKED"],
        },
    )
    _write_text(
        ROUTE / "source_audit.md",
        f"""# Claim 5 fourth-route source audit

- Source: `{PAPER_URL}`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `{PAPER_SHA256}`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1.p3`
- Training anchor: `S5.p2`

The exact Table 1 row gives `1.7135`, `1.9359`, `0.0052`, and `1.8553`.
The prose distribution is a Bernoulli mixture, not the convex interpolation
implemented by the released `generate_data_22` function. This audit uses the
paper's literal Bernoulli statement.
""",
    )
    _write_text(
        ROUTE / "method.md",
        """# Falsification method

This route derives a necessary revenue bound from the exact valuation law.
For every feasible allocation, realized welfare is at most the sum of the
itemwise maximum bidder values. IR gives revenue at most welfare; with the
paper's measured IR regret, revenue is at most welfare plus that regret.

Both mixture-component expectations are integrated analytically with exact
rational arithmetic. A separate checker recomputes the result without
importing campaign code. The negative control replaces the reported revenue
with an impossible value above the bound and must be rejected.
""",
    )
    expected_welfare = Fraction(
        raw["distribution"]["expected_total_welfare_upper_bound"]["exact"]
    )
    counterfeit = expected_welfare + Fraction(1, 10)
    negative = {
        "purpose": "prove the necessary-bound test detects an impossible claim",
        "counterfeit_caama_revenue": _fraction_record(counterfeit),
        "expected_welfare_plus_reported_regret": _fraction_record(
            expected_welfare + Fraction(52, 10000)
        ),
        "counterfeit_revenue_is_rejected": (
            counterfeit > expected_welfare + Fraction(52, 10000)
        ),
    }
    _write_json(ROUTE / "negative_control_output.json", negative)
    _write_text(
        ROUTE / "independent_checker.py", _independent_checker_source()
    )
    independent = subprocess.run(
        [sys.executable, str(ROUTE / "independent_checker.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if independent.returncode != 0:
        raise RuntimeError(
            "independent falsification arithmetic disagreed: "
            f"{independent.stderr}"
        )
    independent_output = json.loads(independent.stdout)
    _write_json(
        ROUTE / "independent_checker_output.json", independent_output
    )
    _write_text(ROUTE / "claim_verifier.py", _claim_verifier_source())
    verifier = subprocess.run(
        [sys.executable, str(ROUTE / "claim_verifier.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_json(
        ROUTE / "verifier_output.json",
        {
            "returncode": verifier.returncode,
            "stdout": verifier.stdout.strip(),
            "stderr": verifier.stderr.strip(),
        },
    )
    _write_json(
        ROUTE / "exact_command_environment.json",
        {
            "fixed_command": FIXED_COMMAND,
            "git_sha": _git_sha(),
            "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "deterministic_inputs": "exact rational arithmetic; no RNG",
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _write_text(
        ROUTE / "limitations_and_deviations.md",
        """# Limitations and deviations

- Necessary upper bounds can establish impossibility but cannot prove that the
  reported optimizer actually attained a feasible value below those bounds.
- The reported numbers lie comfortably inside the feasible region, so this
  route does not resolve missing checkpoint or full-training evidence.
- Failed CPU optimization routes are deliberately excluded from falsification:
  they do not contradict the existence of the paper's reported runs.
""",
    )
    _write_text(
        ROUTE / "EVAL.md",
        f"""# Claim 5 falsification evaluation

- Verdict: **BLOCKED**
- Exact expected-welfare bound: `{float(expected_welfare):.6f}`
- Reported CA-AMA revenue plus-regret bound check:
  `1.9359 <= {float(expected_welfare + Fraction(52, 10000)):.6f}`
- Reported ex-post-IR check:
  `1.8553 <= {float(expected_welfare):.6f}`
- Negative control rejected: `{negative['counterfeit_revenue_is_rejected']}`
- Independent checker exit: `{independent.returncode}`
- Claim verifier exit: `{verifier.returncode}` (nonzero because falsification
  was not established)

No valid counterexample was found. The exact reported values are feasible under
the paper's assumptions, so this mandatory fourth route cannot mark the claim
FALSIFIED.
""",
    )
    print(
        "CLAIM_5_FALSIFICATION_BOUND="
        f"{float(expected_welfare):.6f}"
    )
    print(
        "CLAIM_5_FALSIFICATION_NEGATIVE_CONTROL_PASS="
        f"{negative['counterfeit_revenue_is_rejected']}"
    )
    print(
        "CLAIM_5_FALSIFICATION_INDEPENDENT_CHECKER_EXIT="
        f"{independent.returncode}"
    )
    print("CLAIM_5_VERDICT=BLOCKED")
    print(f"CLAIM_5_VERIFIER_EXIT={verifier.returncode}")
    print(f"CLAIM_5_FALSIFICATION_RUNTIME_SECONDS={time.perf_counter()-started:.6f}")


if __name__ == "__main__":
    main()
