# Correlation-aware auction payments: a claim-by-claim CPU reproduction

![Paper and observed Claim 4 revenues](images/claim4_headline.svg)

The paper asks whether an affine-maximizer auction can use correlation between
bidders without sacrificing dominant-strategy truthfulness. The central idea is
simple: add a payment that depends on rival values but not on the bidder's own
report. The reproduction found strong numerical support for that idea at the
paper's full 3-bidder × 10-item Dirichlet scale. It also found a literal
one-bidder counterexample to two “any number of bidders” theorem statements.

This is an evidence report, not a judge result. The previous live score remains
**3/10** until the external judge evaluates a future Hugging Face revision.

## Evidence at a glance

| Claim | Paper result | Observed evidence | Assessment |
|---|---|---|---|
| 1. Deterministic AMA can be arbitrarily poor for any bidder count | For every \(n\) and \(\epsilon>0\), some \(F\) has \(\mathrm{REV}^{D\text{-}AMA}\le\epsilon\mathrm{REV}\) | The construction is certified for every \(n\ge2\), but for \(n=1\), D-AMA implements the optimal posted price and its ratio is 1 for every positive-revenue distribution | **FALSIFIED**, HIGH |
| 2. Independence equality and correlated separation for any bidder count | D-CA = D-AMA under independence; correlation permits full extraction while AMA is arbitrarily poor | Independence equality remains supported. The correlated “any \(n\)” half has the same valid \(n=1\) counterexample | **FALSIFIED**, HIGH |
| 3. Rival-only \(p_i^{Cor}(V_{-i})\) preserves DSIC | Truthful-versus-misreport utility differences are unchanged | Exact cancellation plus 800 multi-item misreports; an own-bid-dependent negative control exhibits a profitable deviation | **VERIFIED**, HIGH |
| 4. Dirichlet Value Share, \(\alpha=0.5\), 3 × 10 | AMA 3.1363; CA-AMA 3.6205; IR regret 0.0031; ex-post revenue 3.5623 | 3.0530; 3.7359; 0.00281; 3.6863 across five seeds | **BLOCKED**, MEDIUM |
| 5. Linear Mixture Asymmetric, \(\alpha=0.6\), 2 × 5 | AMA 1.7135; CA-AMA 1.9359; IR regret 0.0052; ex-post revenue 1.8553 | CPU-upgrade pilot: 1.4808; 1.5128; 0.00557; 1.4715. Exact feasibility bounds find no contradiction | **BLOCKED**, LOW |

The Table 1 value for Claim 5 regret is **0.0052**. The earlier judge summary's
“near 0.001” is less precise than the source table.

## What changed in the mechanism

For an ordinary affine-maximizer auction, bidder \(i\)'s utility is its allocated
value minus the pivot payment. CA-AMA adds

\[
p_i(v)=p_i^{AMA}(v)+p_i^{Cor}(v_{-i}).
\]

The extra term cancels from the truthful-versus-misreport comparison because
the rivals are unchanged:

\[
[u^{AMA}(v_i)-p_i^{Cor}(v_{-i})]
-
[u^{AMA}(b_i)-p_i^{Cor}(v_{-i})]
=
u^{AMA}(v_i)-u^{AMA}(b_i).
\]

The implementation tests this twice: an exact structural identity for arbitrary
rival-only functions and randomized feasible multi-item menus. The negative
control replaces the rival-only term with one depending on the bidder's own
report; it produces a utility gain of 1.0 and is rejected.

## The literal theorem scope

![The n>=2 construction and the n=1 obstruction](images/theory_scope.svg)

For \(n\ge2\), 30 high-precision certificates cover bidder counts
2, 3, 5, 10, and 50 and requested factors from 0.5 to 0.001. The common
construction chooses
\(\eta=e^{-2/\delta}\) and \(\eta_1=\eta^2\), yielding

\[
\frac{\mathrm{REV}^{D\text{-}AMA}}{\mathrm{REV}}
\le \delta/2+\eta(1-\eta)\delta/2 < \delta.
\]

The source, however, says “any number of bidders.” With one bidder, every
DSIC/IR single-item allocation rule is a mixture of posted-price thresholds:

\[
\mathbb{E}[p(V)]
=\int r\,\Pr[V\ge r]\,dG(r)
\le \sup_r r\,\Pr[V\ge r].
\]

A one-bidder deterministic AMA implements every posted price, so
\(\mathrm{REV}^{D\text{-}AMA}=\mathrm{REV}\) whenever optimal revenue is
positive. This contradicts the claimed factor for any \(\epsilon<1\). A
zero-revenue distribution would make \(0\le\epsilon 0\) vacuously true; the
paper's “arbitrarily small fraction” language and this campaign's non-vacuity
rule require the positive-revenue reading.

The appendix has a second source discrepancy: it defines
\(v_2=\eta_1(1-v_1)\) but prints \(1-v_2/\eta\). Full extraction requires the
inverse \(1-v_2/\eta_1=v_1\).

## Claim 4 at full distribution scale

The full-scale route samples the literal Dirichlet Value Share distribution:
three bidders, ten additive items, and concentration \(\alpha=0.5\). It trains a
separate three-linear-layer ReLU payment network for each bidder, consuming only
the other two bidders' 20 values. Five independent training seeds use 2,000
updates each. A disjoint 10,000-profile validation set chooses a global payment
scale satisfying the paper's IR-regret target, and a fixed 20,000-profile hard
test evaluates each seed.

![Claim 4 seed stability and IR regret](images/claim4_seed_stability.svg)

| Metric | Paper | Observed mean | Seed-level 95% interval |
|---|---:|---:|---:|
| Randomized AMA revenue | 3.1363 | 3.0530 | fixed common test set |
| CA-AMA revenue | 3.6205 | 3.7359 | [3.7297, 3.7422] |
| CA-AMA gain | 0.4842 | 0.6829 | [0.6767, 0.6892] |
| Ex-post IR regret | 0.0031 | 0.00281 | [0.00273, 0.00289] |
| Ex-post IR revenue | 3.5623 | 3.6863 | [3.6807, 3.6918] |

All numerical tolerances pass. The independent checker re-reads 100,000 raw
profiles, verifies the seed/sample accounting, and checks samplewise
revenue \(\le\) welfare + IR regret. Reversing rival profiles raises regret
from 0.00281 to 0.12943, so the learned payment is using the intended
correlation.

Why **BLOCKED** rather than VERIFIED? The released materials omit a learned
3 × 10, 2048-menu AMenuNet checkpoint and publish a contradictory 10 × 3
command. This route uses a validated separable reserve allocation core and a
held-out scalar calibration. Those are material substitutions even though the
distribution, rival-only payment architecture, scale, seed count, uncertainty,
and numeric effect are direct.

## Claim 5: four routes, no valid verdict upgrade

![Claim 5 paper values, CPU pilot, and exact welfare bound](images/claim5_context.svg)

The paper describes a Bernoulli mixture: independently per item,
\(v_1\sim U[0,1]\); with probability 0.6,
\(v_2=(1-v_1)/4\), otherwise \(v_2\sim U[0,1/4]\). The released
`generate_data_22` instead uses convex interpolation. The reproduction follows
the paper text.

| Route | Method | Outcome |
|---|---|---|
| 1 | Source, data-generator, and code audit | Found the Bernoulli/convex-interpolation mismatch |
| 2 | Vectorized paper-semantics optimizer | Undertrained pilot; faithful full CPU training projected impractical |
| 3 | Direct mechanism-space `cpu-upgrade` run | 1.4808 baseline, 1.5128 CA, regret 0.00557; did not reach the paper values |
| 4 | Mandatory exact falsification route | Expected welfare bound \(623/240=2.59583\); all reported values are feasible |

The fourth route uses exact rational arithmetic. It verifies
\(1.9359\le2.59583+0.0052\) and
\(1.8553\le2.59583\). A counterfeit revenue above the bound is rejected.
Because no assumption-satisfying counterexample was found and full faithful
optimization remains unavailable, Claim 5 is **BLOCKED**, not falsified.

## Reproducibility and provenance

All scientific nodes inherit the exact command:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```

The winning cumulative run is
[`orx/cumulative-evidence-parser-fix`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/cumulative-evidence-parser-fix)
at `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`. It ran on local CPU for
14m32s, passed 25 tests, and produced 89 SHA-256-manifested artifacts. Local
CPU has no incremental cloud charge. The Claim 5 direct pilot used Hugging
Face `cpu-upgrade` for 1h58m; the orchestration record does not expose a billed
amount, so no cost is invented.

Important experiment branches:

- [Frozen baseline](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/frozen-judged-reproduction-baseline)
- [Literal theorem-scope audit](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/literal-n-1-theorem-scope-audit)
- [Claim 4 five-seed validation](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/claim-4-neural-multi-seed-validation)
- [Claim 5 direct CPU-upgrade pilot](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/direct-mechanism-space-claim-5-reproduction)
- [Claim 5 falsification audit](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/claim-5-mandatory-falsification-audit)

Machine-readable entry points:

- [Cumulative verdict index](../../.openresearch/artifacts/campaign_verdicts.json)
- [SHA-256 artifact manifest](../../.openresearch/artifacts/campaign_manifest.json)
- [Claim 1 evaluation](../../.openresearch/artifacts/claim_1/EVAL.md)
- [Claim 4 multi-seed evaluation](../../.openresearch/artifacts/claim_4/route_3_cross_item_pcor_multiseed/EVAL.md)
- [Claim 5 falsification evaluation](../../.openresearch/artifacts/claim_5/route_4_falsification_audit/EVAL.md)

## Assessment

The reproduction materially improves the evidence beyond the prior toy-only
state: the theory claims are resolved at their literal quantifiers, the DSIC
claim has exact structural evidence, and Claim 4 has a stable full-scale
five-seed effect. It does not honestly close Claims 4 or 5 under the paper's
exact released training procedure. Those remain BLOCKED for missing or
inconsistent public artifacts and CPU-feasibility limits.
