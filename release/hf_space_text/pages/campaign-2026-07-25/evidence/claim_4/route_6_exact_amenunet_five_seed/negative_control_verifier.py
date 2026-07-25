#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
independent = json.loads((here / "independent_checker_output.json").read_text())
checks = {
    "zero_pcor_ablation_removes_revenue_ci": independent["checks"][
        "zero_pcor_ablation_removes_revenue_ci"
    ],
    "rival_reversal_increases_regret_ci": independent["checks"][
        "rival_reversal_increases_regret_ci"
    ],
}
result = {"checks": checks, "all_checks_pass": all(checks.values())}
print(json.dumps(result, sort_keys=True))
sys.exit(0 if result["all_checks_pass"] else 1)
