# Claim 1 — literal scope falsification

**Verdict: FALSIFIED (HIGH confidence).**

The source says that for “any number of bidders `n`” and every positive
approximation factor, there exists a distribution where deterministic AMA is
an arbitrarily small fraction of optimal revenue.

For every `n>=2`, 30 high-precision certificates validate the paper's intended
construction. For `n=1`, every DSIC/IR single-item mechanism is a mixture of
posted-price thresholds:

`E[p(V)] = ∫ r Pr[V>=r] dG(r) <= sup_r r Pr[V>=r]`.

A one-bidder deterministic AMA implements every posted price, so its ratio is
1 for every distribution with positive optimal revenue. This contradicts the
literal claim for any factor below one.

A zero-revenue distribution would satisfy the displayed inequality vacuously.
The paper's “arbitrarily small fraction” language and this campaign's
non-vacuity rule require positive revenue.

Evidence: [contract](../evidence/claim_1/claim_contract.json) ·
[source audit](../evidence/claim_1/source_audit.md) ·
[raw results](../evidence/claim_1/raw_results.json) ·
[verifier output](../evidence/claim_1/verifier_output.json).
