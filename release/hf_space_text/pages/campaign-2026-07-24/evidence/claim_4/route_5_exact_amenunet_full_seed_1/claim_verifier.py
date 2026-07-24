#!/usr/bin/env python3
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
independent = json.loads((here / "independent_checker_output.json").read_text())
config = json.loads((here / "config.json").read_text())
ok = (
    independent["route_integrity_pass"]
    and independent["raw_rows"] == 20000
    and config["baseline_updates"] == 32000
    and config["mutual_updates"] == 16000
    and config["post_updates"] == 16000
    and independent["claim_verdict"] == "BLOCKED"
    and not independent["claim_verified"]
)
print(json.dumps({
    "claim": 4,
    "route": "exact_amenunet_full_seed_1",
    "route_integrity_pass": ok,
    "verdict": "BLOCKED",
    "reason": "one of five required training seeds",
}, sort_keys=True))
sys.exit(0 if ok else 1)
