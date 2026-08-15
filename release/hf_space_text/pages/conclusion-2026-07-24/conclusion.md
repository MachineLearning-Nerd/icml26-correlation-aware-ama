# Conclusion

---
<!-- trackio-cell
{"type":"artifact","id":"cell_reproduction_bundle_20260724","created_at":"2026-07-24T00:06:00+00:00","title":"Reproduction bundle"}
-->
Reproduction code and text evidence:
https://github.com/MachineLearning-Nerd/icml26-correlation-aware-ama

---
<!-- trackio-cell
{"type":"markdown","id":"cell_rerun_20260724","created_at":"2026-07-24T00:06:01+00:00","title":"Download and rerun"}
-->
Clone the repository, install the locked Python 3.12 environment with
`uv sync --frozen`, then run:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```

The release manifest identifies the winning Git SHA, exact text-file upload
allowlist, deterministic seeds, raw evidence paths, and verifier exit codes.
