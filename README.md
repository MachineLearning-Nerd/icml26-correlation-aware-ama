# ICML 2026 — CA-AMA: Correlation-Aware Payment

Independent reproduction audit for *Enhancing Affine Maximizer Auctions with
Correlation-Aware Payment*.

Paper: [arXiv:2602.09455](https://arxiv.org/abs/2602.09455) ·
[OpenReview: TA3NDHgNJh](https://openreview.net/forum?id=TA3NDHgNJh) ·
[official implementation](https://github.com/Haoran0301/CA-AMA)

Repository: `icml26-correlation-aware-ama` · owner: `MachineLearning-Nerd`

## Current verdict

Overall status: **INCONCLUSIVE — VERIFIED SCOPED FINDINGS WITH BLOCKED
PAPER-LEVEL EMPIRICAL CLAIMS**.

| Claim | Verdict | What is actually established |
| --- | --- | --- |
| C1 | `FALSIFIED_LITERALLY` | The paper says “any number of bidders,” but (n=1) is a counterexample. The intended (n\ge2) construction is supported by high-precision certificates. |
| C2 | `FALSIFIED_LITERALLY` | The intended independence identity and multi-bidder separation are supported, but the literal one-bidder scope cannot exhibit correlation-based separation. |
| C3 | `VERIFIED` | Exact symbolic cancellation, exhaustive finite checks, and an own-report negative control support the rival-only DSIC argument. |
| C4 | `BLOCKED` | The exact released 3-bidder × 10-item AMenuNet seed-1 run is close to the paper, but the exact five-seed paper aggregate is not available as a complete released-core reproduction. |
| C5 | `BLOCKED` | Four routes found a paper/code data-contract mismatch, undertrained pilots, and no valid falsification; a paper-faithful full optimization remains unavailable. |

The previous live judge score recorded in the repository is **3/10**. No new
judge score or author endorsement is claimed here. The historical Hugging Face
logbook is [DineshAI/TA3NDHgNJh](https://huggingface.co/spaces/DineshAI/TA3NDHgNJh).

## What the paper does

CA-AMA augments an affine maximizer auction with a correlation-aware payment
term. The term for bidder (i) depends on rival values (v_{-i}), not on the
bidder's own report, so it cancels from truthful-versus-misreported utility
differences and preserves dominant-strategy incentive compatibility. The
paper optimizes revenue under individual-rationality constraints, gives a
continuity/generalization discussion for the learned payment, and evaluates
Dirichlet Value Share and Linear Mixture Asymmetric distributions.

This repository separates exact theorem checks, mechanism checks, empirical
reproduction attempts, negative controls, and historical publication material.
A blocked or falsified status is a result of the audit, not a claim about the
authors' intent.

## Claim-to-evidence map

The production paths and evidence boundaries are documented in
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md).

### C1 — deterministic AMA can be arbitrarily poor

`repro/src/theory_campaign.py::run_claim_1` produces the quantified
(n\ge2) certificates and the literal (n=1) counterexample. The claim is
`FALSIFIED_LITERALLY`; the intended multi-bidder construction is retained as
scoped supporting evidence.

### C2 — independence equality and correlated separation

`repro/src/theory_campaign.py::run_claim_2` produces the independence and
correlation certificates plus the one-bidder scope audit. It is
`FALSIFIED_LITERALLY` for the printed “any (n)” wording, while the intended
components are supported.

### C3 — rival-only payment preserves DSIC

`repro/src/theory_campaign.py::run_claim_3` produces exact cancellation and
finite multi-item property checks. The own-report-dependent payment negative
control creates a profitable deviation. This claim is `VERIFIED` within the
stated rival-only mechanism property.

### C4 — Dirichlet Value Share experiment

The exact released AMenuNet seed-1 route reproduces the 3 × 10 setting and
reports baseline revenue `3.090107` and CA-AMA revenue `3.567311`, versus the
paper's `3.1363` and `3.6205`. The five-seed pCor proxy passes its numeric
checks, but `exact_released_2048_menu_core_available=false`; therefore the
paper-level claim remains `BLOCKED`.

### C5 — Linear Mixture Asymmetric experiment

The four routes audit the source/data contract, run a vectorized
paper-semantics pilot, run a direct mechanism-space pilot, and search for an
exact analytical falsification. The paper's Bernoulli mixture and released
`generate_data_22` convex interpolation differ. No valid counterexample was
found, so the claim remains `BLOCKED`.

## Reproduce the current audit

Use Python 3.12, as required by `pyproject.toml`:

```bash
uv sync --frozen --python 3.12
uv run --frozen python repro/src/run_caama.py
uv run --frozen python -m pytest -q repro/tests
```

The scientific command can be CPU-intensive, especially the exact neural
route. The publication checks are lightweight and do not launch training:

```bash
uv run --frozen python repro/src/verify_publication.py
uv run --frozen python repro/src/publication_gate.py --skip-producers
```

The historical campaign evidence is durable under
[`.openresearch/artifacts/`](.openresearch/artifacts/); the current gate never
promotes a proxy or an undertrained run to an exact paper-level result.

## Repository layout

- `repro/src/` — clean-room theory, mechanism, empirical, and publication checks.
- `repro/tests/` — tests for the executable reproduction modules.
- `.openresearch/artifacts/` — claim contracts, raw outputs, controls, and
  historical campaign evidence.
- `upstream/` — copied official implementation snapshot; the pinned source
  commit is recorded in [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).
- `reports/caama-claim-campaign/` — technical claim-by-claim report.
- `release/hf_space_text/` — historical text-only reader snapshot, sanitized
  for repository publication.
- `STATUS.md` — current verdict boundary and production paths.
- `CLAIM_EVIDENCE.md` — how every claim status is produced.
- `BRANCH_AUDIT.md` — old-to-new branch names, tips, and roles.
- `SOURCE_MANIFEST.md` — paper/source boundaries, citation, and thanks.

## Clean branch map

`main` is the reader-facing publication surface. Former experiment-runner
branches use descriptive `audit/`, `baseline/`, `experiment/`, and `release/`
names. Their detailed old-to-new mapping is in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md).

| Clean branch | Role |
| --- | --- |
| `main` | Publication README, report, evidence, and gate. |
| `baseline/frozen-judged-reproduction` | Frozen historical judged baseline and environment. |
| `audit/literal-theorem-scope` | Literal (n=1) theorem-scope audit. |
| `audit/quantified-theory` | Quantified theorem contracts and certificates. |
| `audit/data-contract-cpu-profile` | Paper/code data-contract and CPU feasibility audit. |
| `audit/claim-5-falsification` | Mandatory Claim 5 falsification route. |
| `audit/cumulative-evidence-parser-fix` | Cumulative evidence parser regression. |
| `experiment/claim-4-conditional-support` | Conditional-support full-scale Claim 4 route. |
| `experiment/claim-4-cross-item-pcor` | Cross-item pCor neural pilot. |
| `experiment/claim-4-amenunet-cpu-feasibility` | Exact AMenuNet CPU feasibility route. |
| `experiment/claim-4-amenunet-seed-1` | Exact released AMenuNet seed-1 route. |
| `experiment/claim-4-five-seed-aggregate` | Pre-registered five-seed aggregate route. |
| `experiment/claim-4-fixed-test-rng` | Fixed test-RNG correction route. |
| `experiment/claim-4-neural-multiseed` | Neural multi-seed pCor validation. |
| `experiment/claim-5-direct-mechanism` | Claim 5 direct mechanism-space pilot. |
| `experiment/claim-5-full-paper-semantics` | Full Claim 5 paper-semantics route. |
| `experiment/claim-5-vectorized-paper-semantics` | Vectorized Claim 5 training pilot. |
| `release/canonical-logbook` | Canonical five-claim logbook candidate. |
| `release/claim-4-exact-seed-evidence` | Exact Claim 4 evidence-only release. |
| `release/claim-4-cumulative-update` | Claim 4 cumulative release update. |
| `release/cumulative-evidence` | Cumulative evidence release candidate. |
| `release/portable-evidence` | Portable cumulative evidence manifest. |
| `release/presentation-candidate` | Presentation/readme release candidate. |

## Citation

```bibtex
@article{sun2026enhancing,
  title={Enhancing Affine Maximizer Auctions with Correlation-Aware Payment},
  author={Sun, Haoran and Xia, Xuanzhi and Chu, Xu and Deng, Xiaotie},
  journal={arXiv preprint arXiv:2602.09455},
  year={2026},
  url={https://arxiv.org/abs/2602.09455}
}
```

## Thank you

Thank you to Haoran Sun, Xuanzhi Xia, Xu Chu, and Xiaotie Deng for making the
CA-AMA paper and official implementation available for independent study. The
paper provided a useful basis for testing mechanism-level and empirical
reproducibility. This repository is an independent audit and does not
represent the authors' implementation, approval, or endorsement.

## Related files

- [Technical claim report](reports/caama-claim-campaign/report.md)
- [Current status](STATUS.md)
- [Claim evidence production paths](CLAIM_EVIDENCE.md)
- [Branch audit](BRANCH_AUDIT.md)
- [Source manifest](SOURCE_MANIFEST.md)
