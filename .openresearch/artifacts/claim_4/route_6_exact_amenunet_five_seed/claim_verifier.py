#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
independent = json.loads((here / "independent_checker_output.json").read_text())
negative = json.loads((here / "negative_control_verifier_output.json").read_text())
checks = dict(independent["checks"])
checks["independent_checker_passed"] = (
    independent["all_checks_pass"]
    and independent["raw_rows"] == 100000
)
checks["negative_control_verifier_passed"] = (
    negative["returncode"] == 0
    and negative["all_checks_pass"]
)
ok = all(checks.values())
verdict = "VERIFIED" if ok else "BLOCKED"
print(json.dumps({
    "claim": 4,
    "verdict": verdict,
    "ok": ok,
    "checks": checks,
}, sort_keys=True))
sys.exit(0 if ok else 1)
