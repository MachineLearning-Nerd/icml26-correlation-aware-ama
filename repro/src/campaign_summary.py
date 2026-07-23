#!/usr/bin/env python3
"""Build the fail-closed cumulative verdict index and evidence manifest."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".openresearch" / "artifacts"
FIXED_COMMAND = (
    "uv run --frozen python repro/src/run_caama.py && "
    "uv run --frozen python -m pytest -q repro/tests"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


def _checks_payload(record: dict[str, Any]) -> dict[str, bool]:
    """Accept the two explicit verifier record formats used by the campaign."""
    if "checks" in record:
        return record["checks"]
    if "stdout" in record:
        return json.loads(record["stdout"])["checks"]
    raise KeyError("verifier record contains neither checks nor stdout")


def _theory_verdict(claim: int) -> tuple[str, bool]:
    record = _read_json(
        ARTIFACT_ROOT / f"claim_{claim}" / "verifier_output.json"
    )
    output = json.loads(record["stdout"])
    return output["verdict"], record["returncode"] == 0 and output["ok"]


def main() -> None:
    theory = {
        claim: _theory_verdict(claim)
        for claim in (1, 2, 3)
    }
    claim4_route2 = _read_json(
        ARTIFACT_ROOT
        / "claim_4"
        / "route_2_conditional_support"
        / "verifier_output.json"
    )
    claim4_route3 = _read_json(
        ARTIFACT_ROOT
        / "claim_4"
        / "route_3_cross_item_pcor_multiseed"
        / "claim_verifier_output.json"
    )
    claim4_independent = _read_json(
        ARTIFACT_ROOT
        / "claim_4"
        / "route_3_cross_item_pcor_multiseed"
        / "independent_checker_output.json"
    )
    claim4_summary = _read_json(
        ARTIFACT_ROOT
        / "claim_4"
        / "route_3_cross_item_pcor_multiseed"
        / "aggregate_summary.json"
    )
    claim5_route4 = _read_json(
        ARTIFACT_ROOT
        / "claim_5"
        / "route_4_falsification_audit"
        / "verifier_output.json"
    )
    claim5_independent = _read_json(
        ARTIFACT_ROOT
        / "claim_5"
        / "route_4_falsification_audit"
        / "independent_checker_output.json"
    )

    route_inventory = {
        "claim_4": [
            {
                "route": 1,
                "method": "paper/source/data contract and released-code audit",
                "run_id": "eec88b63-1127-4fde-9a36-2b76223a89e4",
                "result": "full 3x10 distribution confirmed; released 10x3 command conflicts with Table 1",
            },
            {
                "route": 2,
                "method": "analytic conditional-support full-scale mechanism",
                "run_id": "5253861a-73f7-498d-abdb-5226b6297628",
                "result": "positive gain but neural 2048-menu procedure and target magnitude unresolved",
            },
            {
                "route": 3,
                "method": "five-seed rival-only three-layer neural pCor",
                "run_id": "8770c5f1-7f57-4383-8caf-c69eb475714c",
                "result": "all numeric tolerances pass; exact released 3x10 core remains unavailable",
            },
        ],
        "claim_5": [
            {
                "route": 1,
                "method": "literal Bernoulli-mixture source/data audit",
                "run_id": "eec88b63-1127-4fde-9a36-2b76223a89e4",
                "result": "paper distribution differs from released convex interpolation",
            },
            {
                "route": 2,
                "method": "vectorized paper-semantics optimization",
                "run_id": "183ee2a0-a7a7-4592-b7e7-e0fc3595bf9d",
                "result": "undertrained pilot; full local/HF jobs projected impractical CPU time",
            },
            {
                "route": 3,
                "method": "direct mechanism-space CPU-upgrade optimization",
                "run_id": "233ef5c5-063e-40d8-a810-ef661f153826",
                "result": "1.480823 baseline, 1.512781 CA, 0.005571 regret; pilot underfits paper",
            },
            {
                "route": 4,
                "method": "exact rational falsification bound",
                "run_id": "298cd9b9-0ac9-472b-b929-d56a9ac3613b",
                "result": "reported values satisfy necessary welfare/IR bounds; no valid falsification",
            },
        ],
    }

    claim4_numeric_checks = _checks_payload(claim4_route3)
    claim4_resolved = bool(
        claim4_route2["returncode"] == 1
        and all(claim4_independent["checks"].values())
        and all(
            value
            for key, value in claim4_numeric_checks.items()
            if key != "exact_released_2048_menu_core_available"
        )
        and not claim4_numeric_checks[
            "exact_released_2048_menu_core_available"
        ]
    )
    claim5_output = json.loads(claim5_route4["stdout"])
    claim5_resolved = bool(
        claim5_route4["returncode"] == 1
        and claim5_output["route_integrity"]
        and not claim5_output["valid_falsification_found"]
        and all(claim5_independent["checks"].values())
        and len(route_inventory["claim_5"]) == 4
    )

    verdicts = {
        "paper": "arXiv:2602.09455",
        "git_sha": _git_sha(),
        "fixed_command": FIXED_COMMAND,
        "claims": {
            "1": {
                "verdict": theory[1][0],
                "confidence": "HIGH",
                "resolved": theory[1][1],
                "basis": "literal n=1 positive-revenue counterexample plus valid n>=2 certificates",
            },
            "2": {
                "verdict": theory[2][0],
                "confidence": "HIGH",
                "resolved": theory[2][1],
                "basis": "independence equality supported; literal any-n correlated separation falsified at n=1",
            },
            "3": {
                "verdict": theory[3][0],
                "confidence": "HIGH",
                "resolved": theory[3][1],
                "basis": "exact rival-only cancellation identity and multi-item property checks",
            },
            "4": {
                "verdict": "BLOCKED",
                "confidence": "MEDIUM",
                "resolved": claim4_resolved,
                "basis": (
                    "five-seed full-scale numbers pass, but the exact released "
                    "3x10 learned 2048-menu AMA core/checkpoint is unavailable"
                ),
                "observed": {
                    key: claim4_summary[key]["mean"]
                    for key in (
                        "baseline_revenue",
                        "caama_revenue",
                        "caama_ir_regret",
                        "caama_ex_post_ir_revenue",
                    )
                },
            },
            "5": {
                "verdict": "BLOCKED",
                "confidence": "LOW",
                "resolved": claim5_resolved,
                "basis": (
                    "four routes completed; exact falsification found no "
                    "contradiction and faithful full optimization remains unavailable"
                ),
            },
        },
        "route_inventory": route_inventory,
    }
    all_resolved = all(
        record["resolved"] for record in verdicts["claims"].values()
    )
    verdicts["all_claims_have_honest_terminal_status"] = all_resolved
    if not all_resolved:
        raise AssertionError("cumulative campaign has an unresolved claim status")
    _write_json(ARTIFACT_ROOT / "campaign_verdicts.json", verdicts)

    _write_text(
        ARTIFACT_ROOT / "EVAL.md",
        f"""# Cumulative campaign evaluation

- Claim 1: **{theory[1][0]}** (HIGH)
- Claim 2: **{theory[2][0]}** (HIGH)
- Claim 3: **{theory[3][0]}** (HIGH)
- Claim 4: **BLOCKED** (MEDIUM)
- Claim 5: **BLOCKED** (LOW; all four required routes complete)
- Fixed command: `{FIXED_COMMAND}`
- Git SHA: `{_git_sha()}`

`BLOCKED` is an honest terminal evidence status, not a passing result.
No toy or substituted mechanism is promoted to full verification.
""",
    )
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_file() and path.name != "campaign_manifest.json":
            manifest[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(ARTIFACT_ROOT / "campaign_manifest.json", manifest)
    print(
        "CAMPAIGN_VERDICTS="
        + ",".join(
            f"{claim}:{record['verdict']}"
            for claim, record in verdicts["claims"].items()
        )
    )
    print(f"CAMPAIGN_ARTIFACT_FILES={len(manifest)}")
    print("CAMPAIGN_CUMULATIVE_CHECK=PASS")


if __name__ == "__main__":
    main()
