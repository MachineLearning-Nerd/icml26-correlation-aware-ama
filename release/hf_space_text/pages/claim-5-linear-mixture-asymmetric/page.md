# Claim 5: Linear Mixture Asymmetric Table 1

---
<!-- trackio-cell
{"type":"markdown","id":"cell_claim_5_20260724","created_at":"2026-07-24T00:05:00+00:00","title":"Claim 5 evidence"}
-->
**Paper result.** Linear Mixture Asymmetric, alpha `0.6`, 2 bidders × 5 items:
randomized AMA `1.7135`, CA-AMA `1.9359`, IR regret `0.0052`, and ex-post-IR
revenue `1.8553`.

**Assessment: BLOCKED after four routes.**

1. A source/data audit established the paper's Bernoulli-mixture contract and
   the released code's incompatible convex interpolation.
2. A paper-semantics vectorized route established feasibility and scaling
   limits but was undertrained.
3. A direct Hugging Face `cpu-upgrade` pilot obtained baseline `1.480823`,
   CA-AMA `1.512781`, regret `0.005571`, and ex-post revenue `1.471493`.
4. The mandatory falsification route proved the exact welfare bound
   `623/240 = 2.595833`; the paper's numbers satisfy all necessary bounds, so
   no valid counterexample was found.

The pilot does not match the paper's revenues and does not falsify the claim.
No toy, undertrained, or generator-mismatched result is promoted to VERIFIED
or FALSIFIED.
