# Claim 1: deterministic AMA can be arbitrarily poor

---
<!-- trackio-cell
{"type":"markdown","id":"cell_claim_1_20260724","created_at":"2026-07-24T00:01:00+00:00","title":"Claim 1 evidence"}
-->
**Paper statement.** Proposition 3.1 states that for any number of bidders and
every positive epsilon there is a distribution on which deterministic AMA
revenue is at most epsilon times optimal revenue.

**Observed evidence.** The intended construction is reproduced for arbitrary
`n≥2`: the closed-form and numerical integrals agree to `1e-9`, and the
classic-AMA/optimal ratio decreases from `0.3232` at epsilon `0.1` to `0.1438`
at epsilon `0.001`. The literal positive-revenue `n=1` case is a valid
counterexample because a one-bidder deterministic AMA can implement the
optimal posted price, so its revenue ratio cannot be made arbitrarily small.

**Assessment: FALSIFIED as literally quantified; intended `n≥2` construction
supported.** The appendix's displayed inverse relationship also differs from
the construction by an eta subscript; both scope issues are retained in the
source audit.

Run command:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```
