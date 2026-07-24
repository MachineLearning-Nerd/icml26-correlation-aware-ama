# Claim 2 evaluation

- Verdict: **FALSIFIED**
- Contract: Certify both inequalities giving D-CA=D-AMA on independent product supports, and audit the literal all-n correlated construction with D-CA=REV and D-AMA<delta*REV, including n=1.
- Independent checker: `independent finite-support profile enumeration, separate from Decimal correlated checker`
- Negative control: passed if and only if the deliberately invalid construction/mechanism was rejected.
- Git SHA: `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`
- Fixed command: `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests`
- Limitations: see `limitations_and_deviations.md`.
