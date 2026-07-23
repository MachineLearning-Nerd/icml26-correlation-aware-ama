# Claim 5 — four-route audit

![Claim 5 evidence context](../images/claim5_context.svg)

**Verdict: BLOCKED (LOW confidence after all required routes).**

The Table 1 row reports randomized AMA 1.7135, CA-AMA 1.9359, IR regret
0.0052, and ex-post revenue 1.8553. The paper describes a Bernoulli mixture;
the released generator instead uses convex interpolation.

1. Source/data audit established the literal Bernoulli contract.
2. A vectorized paper-semantics pilot was undertrained; full CPU execution was
   projected impractical.
3. A direct `cpu-upgrade` pilot obtained 1.4808 baseline, 1.5128 CA, and
   0.00557 regret, below the paper's revenues.
4. The mandatory falsification route derived the exact expected-welfare bound
   `623/240=2.59583`. Every reported number satisfies the necessary welfare/IR
   inequalities, so no valid counterexample was found.

Evidence: [falsification evaluation](../evidence/claim_5/route_4_falsification_audit/EVAL.md) ·
[exact bound](../evidence/claim_5/route_4_falsification_audit/raw_bound_results.json) ·
[verifier output](../evidence/claim_5/route_4_falsification_audit/verifier_output.json).
