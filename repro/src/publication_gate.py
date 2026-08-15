"""Run the lightweight publication gate; scientific producers stay opt-in."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-producers",
        action="store_true",
        help="run documentation/evidence checks without launching training",
    )
    args = parser.parse_args()
    if not args.skip_producers:
        print("Refusing to launch the expensive scientific producer; pass --skip-producers for the publication gate.")
        return 2

    verifier = subprocess.run(
        [sys.executable, str(ROOT / "repro/src/verify_publication.py"), "--no-write"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        verification = json.loads(verifier.stdout)
    except json.JSONDecodeError:
        verification = {"status": "FAIL", "errors": [verifier.stdout[-2000:]]}
    status = "SCOPED_PASS" if verifier.returncode == 0 and verification.get("status") == "PASS" else "FAIL"
    result = {"status": status, "verification": verification}
    output = ROOT / "outputs/publication_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if status == "SCOPED_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
