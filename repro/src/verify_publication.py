"""Verify the reader-facing publication surface without launching experiments."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL_NAME = "icml26-correlation-aware-ama"
PAPER_ID = "TA3NDHgNJh"
OFFICIAL_COMMIT = "ed2af19ed02c70b58efdf705635981241222d045"
CANONICAL_AUTHOR = "MachineLearning-Nerd"
CANONICAL_EMAIL = "MachineLearning-Nerd@users.noreply.github.com"
EXPECTED_BRANCHES = {
    "main",
    "baseline/frozen-judged-reproduction",
    "audit/literal-theorem-scope",
    "audit/quantified-theory",
    "audit/data-contract-cpu-profile",
    "audit/claim-5-falsification",
    "audit/cumulative-evidence-parser-fix",
    "experiment/claim-4-conditional-support",
    "experiment/claim-4-cross-item-pcor",
    "experiment/claim-4-amenunet-cpu-feasibility",
    "experiment/claim-4-amenunet-seed-1",
    "experiment/claim-4-five-seed-aggregate",
    "experiment/claim-4-fixed-test-rng",
    "experiment/claim-4-neural-multiseed",
    "experiment/claim-5-direct-mechanism",
    "experiment/claim-5-full-paper-semantics",
    "experiment/claim-5-vectorized-paper-semantics",
    "release/canonical-logbook",
    "release/claim-4-exact-seed-evidence",
    "release/claim-4-cumulative-update",
    "release/cumulative-evidence",
    "release/portable-evidence",
    "release/presentation-candidate",
}


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_json(relative_path: str) -> object:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def claim_verifier(relative_path: str) -> dict[str, object]:
    wrapper = load_json(relative_path)
    if not isinstance(wrapper, dict):
        return {}
    stdout = wrapper.get("stdout", "")
    if not isinstance(stdout, str):
        return {}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tracked_text_errors(errors: list[str]) -> None:
    tracked = run_git("ls-files", "-z").split("\0")
    private_path = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)")
    stale_name = "icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue"
    for relative in tracked:
        if not relative or not (ROOT / relative).is_file():
            continue
        if relative == "repro/src/verify_publication.py":
            continue
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        check(not private_path.search(text), f"private absolute path in {relative}", errors)
        check(stale_name not in text, f"stale repository name in {relative}", errors)


def verify() -> dict[str, object]:
    errors: list[str] = []
    required = [
        "README.md",
        "STATUS.md",
        "CLAIM_EVIDENCE.md",
        "BRANCH_AUDIT.md",
        "SOURCE_MANIFEST.md",
        "sources.json",
        "evidence/claim_summary.json",
        ".openresearch/artifacts/campaign_verdicts.json",
        "release/RELEASE_GATE.md",
        "release/PUBLICATION_RECORD.json",
    ]
    for relative in required:
        check((ROOT / relative).is_file(), f"missing required file: {relative}", errors)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "Enhancing Affine Maximizer Auctions with Correlation-Aware Payment",
        "arXiv:2602.09455",
        "Haoran Sun",
        "Xuanzhi Xia",
        "Xu Chu",
        "Xiaotie Deng",
        "Thank you",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        FINAL_NAME,
    ):
        check(phrase in readme, f"README missing required content: {phrase}", errors)

    sources = load_json("sources.json")
    check(isinstance(sources, dict), "sources.json is not an object", errors)
    if isinstance(sources, dict):
        official = sources.get("official_source")
        check(isinstance(official, dict), "official_source is missing", errors)
        if isinstance(official, dict):
            check(official.get("commit") == OFFICIAL_COMMIT, "official source pin changed", errors)
            check(official.get("url") == "https://github.com/Haoran0301/CA-AMA", "official source URL changed", errors)
        check(sources.get("openreview", {}).get("challenge_id") == PAPER_ID, "OpenReview challenge ID changed", errors)

    summary = load_json("evidence/claim_summary.json")
    check(isinstance(summary, dict), "claim summary is not an object", errors)
    if isinstance(summary, dict):
        check(summary.get("repository") == FINAL_NAME, "claim summary has wrong repository name", errors)
        check(summary.get("score_claim") is False, "score claim must remain false", errors)
        check(summary.get("official_author_endorsement") is False, "author endorsement must remain false", errors)

    verdicts = load_json(".openresearch/artifacts/campaign_verdicts.json")
    check(isinstance(verdicts, dict), "campaign verdicts are not an object", errors)
    if isinstance(verdicts, dict):
        claims = verdicts.get("claims", {})
        expected = {"1": "FALSIFIED", "2": "FALSIFIED", "3": "VERIFIED", "4": "BLOCKED", "5": "BLOCKED"}
        check(isinstance(claims, dict), "campaign claims are missing", errors)
        if isinstance(claims, dict):
            for number, verdict in expected.items():
                record = claims.get(number, {})
                check(isinstance(record, dict) and record.get("verdict") == verdict, f"campaign verdict mismatch for C{number}", errors)

    for number, expected in ((1, "FALSIFIED"), (2, "FALSIFIED"), (3, "VERIFIED")):
        parsed = claim_verifier(f".openresearch/artifacts/claim_{number}/verifier_output.json")
        check(parsed.get("claim") == number and parsed.get("verdict") == expected and parsed.get("ok") is True, f"C{number} verifier evidence is not terminal", errors)

    c4_loaded = load_json(".openresearch/artifacts/claim_4/route_3_cross_item_pcor_multiseed/claim_verifier_output.json")
    c4 = c4_loaded if isinstance(c4_loaded, dict) else {}
    c4_checks = c4.get("checks", {})
    check(isinstance(c4_checks, dict), "C4 proxy checks are missing", errors)
    if isinstance(c4_checks, dict):
        check(c4_checks.get("five_independent_training_seeds") is True, "C4 five-seed proxy check missing", errors)
        check(c4_checks.get("exact_released_2048_menu_core_available") is False, "C4 exact-core boundary was promoted", errors)
    c4_independent = load_json(".openresearch/artifacts/claim_4/route_3_cross_item_pcor_multiseed/independent_checker_output.json")
    check(isinstance(c4_independent, dict), "C4 independent checker is missing", errors)
    if isinstance(c4_independent, dict):
        checks = c4_independent.get("checks", {})
        check(isinstance(checks, dict) and all(value is True for value in checks.values()), "C4 independent checks are incomplete", errors)

    c5 = claim_verifier(".openresearch/artifacts/claim_5/route_4_falsification_audit/verifier_output.json")
    check(c5.get("claim") == 5 and c5.get("route_integrity") is True and c5.get("valid_falsification_found") is False and c5.get("verdict") == "BLOCKED", "C5 falsification boundary changed", errors)
    c5_independent = load_json(".openresearch/artifacts/claim_5/route_4_falsification_audit/independent_checker_output.json")
    check(isinstance(c5_independent, dict), "C5 independent checker is missing", errors)
    if isinstance(c5_independent, dict):
        checks = c5_independent.get("checks", {})
        check(isinstance(checks, dict) and all(value is True for value in checks.values()), "C5 independent checks are incomplete", errors)

    tracked = run_git("ls-files")
    check(not any(line == ".trackio" or line.startswith(".trackio/") for line in tracked.splitlines()), "tracked .trackio export remains", errors)
    tracked_text_errors(errors)

    clean_refs = set()
    for ref in run_git("for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes/origin").splitlines():
        if ref in {"origin", "origin/HEAD"} or ref.endswith("/HEAD"):
            continue
        if ref.startswith("origin/"):
            clean_refs.add(ref.removeprefix("origin/"))
        elif "/" not in ref:
            clean_refs.add(ref)
    check(clean_refs == EXPECTED_BRANCHES, f"branch surface mismatch: expected {len(EXPECTED_BRANCHES)}, found {len(clean_refs)}", errors)
    check(run_git("branch", "--show-current") == "main", "current branch is not main", errors)
    remote_url = run_git("remote", "get-url", "origin")
    check(FINAL_NAME in remote_url, "origin remote still uses the old repository name", errors)

    branch_audit = (ROOT / "BRANCH_AUDIT.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `[^|]+` \| `[^|]+` \|", branch_audit, flags=re.MULTILINE)
    check(len(rows) == len(EXPECTED_BRANCHES), f"branch audit has {len(rows)} data rows", errors)
    for branch in EXPECTED_BRANCHES:
        check(f"`{branch}`" in branch_audit, f"branch audit missing {branch}", errors)

    identities = run_git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").splitlines()
    for identity in identities:
        fields = identity.split("\t")
        check(len(fields) == 4 and fields[0] == CANONICAL_AUTHOR and fields[1] == CANONICAL_EMAIL and fields[2] == CANONICAL_AUTHOR and fields[3] == CANONICAL_EMAIL, f"non-canonical commit identity: {identity}", errors)

    result: dict[str, object] = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="do not write outputs/verification.json")
    args = parser.parse_args()
    result = verify()
    if not args.no_write:
        output = ROOT / "outputs/verification.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
