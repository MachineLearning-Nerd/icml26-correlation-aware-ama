#!/usr/bin/env python3
import json
from pathlib import Path

here = Path(__file__).resolve().parent
criteria = json.loads((here / "verification_criteria.json").read_text())
independent = json.loads(
    (here / "independent_checker_output.json").read_text()
)
checks = dict(criteria["checks"])
checks["independent_checker"] = all(independent["checks"].values())
checks["method_matches_paper_neural_2048_menu"] = False
ok = all(checks.values())
print(json.dumps({
    "claim": 4,
    "verdict": "VERIFIED" if ok else "BLOCKED",
    "checks": checks,
}, sort_keys=True))
raise SystemExit(0 if ok else 1)
