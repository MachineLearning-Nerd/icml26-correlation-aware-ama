#!/usr/bin/env python3
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
