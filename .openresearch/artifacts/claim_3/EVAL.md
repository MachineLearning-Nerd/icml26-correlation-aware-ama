# Claim 3 evaluation

- Verdict: **VERIFIED**
- Contract: Prove the correlation-payment term cancels exactly from the truthful versus misreport utility difference and independently test multi-item menus.
- Independent checker: `fresh random feasible multi-item menus with weighted VCG pivot payments`
- Negative control: passed if and only if the deliberately invalid construction/mechanism was rejected.
- Git SHA: `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`
- Fixed command: `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests`
- Limitations: see `limitations_and_deviations.md`.
