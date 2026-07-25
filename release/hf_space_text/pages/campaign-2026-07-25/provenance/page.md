# Claim 4 verification provenance

Scientific branch:
`orx/claim-4-exact-five-seed-parallel-verification`.

Evidence Git SHA:
`8b3aa42e97f01f1202a7bed4b38394c6be88e6f0`.

Run:
`3deb95be-0518-43a4-a802-d2e19ad5c63d`.

Fixed command:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```

Environment: repository-level `uv` lock, Python 3.12.11, PyTorch 2.7.1 CPU,
8 logical/physical ARM64 cores, no CUDA. The five workers used two PyTorch
threads each. Wall time was 51,470 seconds (14h21m).

The source HTML was retrieved with an explicit User-Agent and has SHA-256
`2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`.

The public 10×3 command conflicts with the paper's 3×10 table. Menu size 2,048
is inferred from that public row, and the payment-network description differs
between paper and code. These limitations are retained in the formal evidence.
