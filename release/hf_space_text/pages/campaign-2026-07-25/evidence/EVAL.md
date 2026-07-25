# Cumulative campaign evaluation

- Claim 1: **FALSIFIED** (HIGH)
- Claim 2: **FALSIFIED** (HIGH)
- Claim 3: **VERIFIED** (HIGH)
- Claim 4: **VERIFIED** (HIGH)
- Claim 5: **BLOCKED** (LOW; all four required routes complete)
- Fixed command: `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests`
- Git SHA: `c42ef4d257d5a44ac8d84634a1b51c9161aad1b5`

Claim 4 is promoted only from the exact five-seed verifier. Claim 5 remains
honestly BLOCKED; no toy or substituted mechanism is promoted.
