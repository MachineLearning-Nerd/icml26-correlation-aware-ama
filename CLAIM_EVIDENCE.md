# Claim evidence production paths

This file explains how each paper claim is turned into a repository verdict.
The distinction is deliberate: a paper result, a clean-room check, a proxy,
and an unresolved reproduction attempt are not interchangeable evidence.

## C1 — deterministic AMA can be arbitrarily poor

- Producer: `repro/src/theory_campaign.py::run_claim_1`.
- Durable evidence: `.openresearch/artifacts/claim_1/`.
- Independent checks: `verifier_output.json`,
  `independent_checker_output.json`, and `negative_control_output.json`.
- Result: high-precision certificates support the intended construction for
  (n\ge2), while the exact (n=1) posted-price counterexample contradicts the
  literal “any number of bidders” wording.
- Promoted status: `FALSIFIED_LITERALLY`; the scoped (n\ge2) construction is
  reported as supporting evidence, not as a literal full-claim pass.

## C2 — independence equality and correlated separation

- Producer: `repro/src/theory_campaign.py::run_claim_2`.
- Durable evidence: `.openresearch/artifacts/claim_2/`.
- Independent checks: exact independence/correlation certificates and the
  negative control in that directory.
- Result: intended multi-bidder identities pass, but a one-bidder market has no
  rival profile and cannot realize the stated correlated separation.
- Promoted status: `FALSIFIED_LITERALLY`.

## C3 — rival-only payment preserves DSIC

- Producer: `repro/src/theory_campaign.py::run_claim_3`.
- Durable evidence: `.openresearch/artifacts/claim_3/`.
- Mechanism: compare truthful and misreported utilities with
  (p_i^{Cor}(v_{-i})) held fixed; the added term cancels exactly.
- Independent checks: symbolic cancellation, exhaustive finite checks across
  2–4 bidders and 1–3 items, and an own-report-dependent negative control that
  finds a profitable deviation.
- Promoted status: `VERIFIED` for the rival-only mechanism property.

## C4 — Dirichlet Value Share experiment

Paper setting: α=`0.5`, 3 bidders, 10 additive items, menu size 2048,
fixed 20,000-profile test set. The paper reports AMA `3.1363`, CA-AMA
`3.6205`, IR regret `0.0031`, and ex-post revenue `3.5623`.

Routes recorded under `.openresearch/artifacts/claim_4/`:

1. `route_1_data_and_profile/` audits the source, distribution, and profile
   contract.
2. `route_2_conditional_support/` tests a conditional-support full-scale
   mechanism; it does not reproduce the released neural training procedure.
3. `route_3_cross_item_pcor_multiseed/` trains a five-seed three-layer pCor
   proxy. Its numeric checks pass, but
   `exact_released_2048_menu_core_available` is `false`.
4. The exact released AMenuNet seed-1 evidence is retained in the release
   snapshot and reports baseline `3.090107`, CA-AMA `3.567311`, IR regret
   `0.006133`, and ex-post revenue `3.466351`.

The exact five-seed released-core aggregate is not available. The pCor proxy
is therefore a useful bounded observation, not a paper-level reproduction.
Promoted status: `BLOCKED`.

## C5 — Linear Mixture Asymmetric experiment

Paper setting: α=`0.6`, 2 bidders, 5 additive items. The paper reports AMA
`1.7135`, CA-AMA `1.9359`, IR regret `0.0052`, and ex-post revenue `1.8553`.

Routes recorded under `.openresearch/artifacts/claim_5/`:

1. `route_1_data_and_profile/` finds that the paper describes a Bernoulli
   mixture while released `generate_data_22` uses convex interpolation.
2. The vectorized paper-semantics pilot is faithful in form but undertrained.
3. The direct mechanism-space `cpu-upgrade` pilot reports baseline `1.480823`,
   CA-AMA `1.512781`, IR regret `0.005571`, and ex-post revenue `1.471493`;
   it underfits the paper protocol.
4. `route_4_falsification_audit/` checks the exact welfare/IR bound
   (623/240=2.595833\ldots). The reported values are feasible and no valid
   falsification is found.

A failed or undertrained reproduction is not a falsification. With the
   unresolved public data contract and no assumption-matching counterexample,
   the promoted status is `BLOCKED`.

## Reproduction commands and limits

The historical campaign command is:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```

It can take hours on CPU. The publication verifier and gate intentionally do
not run this producer. No exact paper-level claim is inferred from the
lightweight gate alone.
