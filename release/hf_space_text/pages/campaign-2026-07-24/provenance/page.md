# Campaign provenance and limitations

Winning scientific branch:
`audit/cumulative-evidence-parser-fix`.

Winning Git SHA:
`bf4cc9371feea65edf71ad1dc998ed88de23b7a7`.

Fixed command inherited by every scientific node:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```

Environment: repository-level `uv` lock, Python 3.12, PyTorch 2.7.1 CPU. The
winning cumulative run took 14m32s on local CPU, passed 25 tests, and produced
89 manifested artifacts. The Claim 5 direct pilot took 1h58m on Hugging Face
`cpu-upgrade`; the orchestration record does not expose billed cost.

Source: ar5iv HTML retrieved 2026-07-23 with explicit User-Agent, SHA-256
`2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`.

This candidate is additive to judged Space revision
`1c13494fc9e76a381d76c681cfd582495eb79d02`. Existing pages remain unchanged
and reachable. BLOCKED means that the route sequence is fully documented but
does not establish VERIFIED or FALSIFIED evidence.

Evidence: [campaign index](../evidence/campaign_verdicts.json) ·
[manifest](../evidence/campaign_manifest.json) ·
[cumulative evaluation](../evidence/EVAL.md).
