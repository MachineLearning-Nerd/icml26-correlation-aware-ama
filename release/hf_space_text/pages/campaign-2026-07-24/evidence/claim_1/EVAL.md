# Claim 1 evaluation

- Verdict: **FALSIFIED**
- Contract: Audit every positive integer n. For n>=2, construct F explicitly and certify REV_F^D-AMA < delta*REV_F. For n=1, test whether a positive-revenue separation can exist under DSIC and IR.
- Independent checker: `Decimal recomputation of both the paper bound and its simplified identity`
- Negative control: passed if and only if the deliberately invalid construction/mechanism was rejected.
- Git SHA: `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`
- Fixed command: `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests`
- Limitations: see `limitations_and_deviations.md`.
