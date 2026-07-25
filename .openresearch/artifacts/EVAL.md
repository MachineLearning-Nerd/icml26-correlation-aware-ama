# Cumulative campaign evaluation

- Claim 1: **FALSIFIED** (HIGH)
- Claim 2: **FALSIFIED** (HIGH)
- Claim 3: **VERIFIED** (HIGH)
- Claim 4: **BLOCKED** (MEDIUM)
- Claim 5: **BLOCKED** (LOW; all four required routes complete)
- Fixed command: `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests`
- Git SHA: `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`

`BLOCKED` is an honest terminal evidence status, not a passing result.
No toy or substituted mechanism is promoted to full verification.
