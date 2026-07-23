# OpenResearch claim-by-claim reproduction

This campaign reproduces *Enhancing Affine Maximizer Auctions with
Correlation-Aware Payment* ([arXiv:2602.09455](https://arxiv.org/abs/2602.09455))
on CPU. It tests the paper's theorem quantifiers, the rival-only DSIC payment
argument, and both requested Table 1 settings. The previous live judge score is
still **3/10**; no new points are claimed before another live evaluation.

The strongest empirical result is the full 3-bidder × 10-item Dirichlet
experiment. The paper reports randomized AMA 3.1363 and CA-AMA 3.6205. Five
local-CPU seeds produced 3.0530 and 3.7359, with IR regret 0.00281 versus the
paper's 0.0031. This is substantial support, but the verdict remains **BLOCKED**
because the release lacks a matching learned 3 × 10, 2048-menu core and this
route substitutes a separable reserve core plus held-out payment scaling.

The literal “any number of bidders” wording in Claims 1 and 2 is
**FALSIFIED** at \(n=1\): a one-bidder deterministic AMA implements the optimal
posted price, so its positive-revenue ratio is 1. Claim 3 is **VERIFIED** by
exact payment cancellation and multi-item checks. Claim 5 is **BLOCKED** after
four routes: its `cpu-upgrade` pilot underfits the paper values, while an exact
welfare/IR audit finds no valid counterexample.

Compute was local CPU except for the 1h58m Claim 5 direct pilot on Hugging Face
`cpu-upgrade`; no GPU was used. The environment is the repository-level
Python 3.12 `.venv` locked by `uv`.

[Read the illustrated report](reports/caama-claim-campaign/report.md) ·
[Open the tutorial notebook](notebooks/caama_reproduction.py) ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/blob/main/notebooks/caama_reproduction.py)

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/frozen-judged-reproduction-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/frozen-judged-reproduction-baseline) | Freeze the judged reproduction | `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests` | Existing two-bidder toy baseline preserved | Local CPU, 1m01s |
| [`orx/literal-n-1-theorem-scope-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/literal-n-1-theorem-scope-audit) | Test the exact “any \(n\)” scope | `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests` | Claims 1–2 FALSIFIED literally; intended \(n\ge2\) construction passes | Local CPU, 1m42s |
| [`orx/claim-4-neural-multi-seed-validation`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/claim-4-neural-multi-seed-validation) | Full 3 × 10, five-seed pCor validation | `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests` | Numeric contract passes; exact paper core unavailable, so BLOCKED | Local CPU, 21m36s |
| [`orx/direct-mechanism-space-claim-5-reproduction`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/direct-mechanism-space-claim-5-reproduction) | Claim 5 direct-mechanism pilot | `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests` | 1.4808 baseline, 1.5128 CA; pilot underfits paper | HF `cpu-upgrade`, 1h58m |
| [`orx/claim-5-mandatory-falsification-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/claim-5-mandatory-falsification-audit) | Exact fourth-route counterexample search | `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests` | No valid falsification; Claim 5 BLOCKED | Local CPU, 5m25s |
| [`orx/cumulative-evidence-parser-fix`](https://github.com/MachineLearning-Nerd/icml26-repro-TA3NDHgNJh-ca-ama-correlated-revenue/tree/orx/cumulative-evidence-parser-fix) | Single-SHA cumulative evidence regression | `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests` | 25 tests; 89 manifested artifacts; winning evidence SHA `bf4cc93` | Local CPU, 14m32s |
| `main` | Public README, report, notebook, and evidence surface | Not run as an experiment (publication surface) | Presentation-only; publish after explicit approval | No experiment compute |

The full report records substitutions, uncertainty, negative controls, source
discrepancies, all four Claim 5 routes, and links to machine-readable evidence.

---

## Legacy judged baseline (preserved for provenance)

The section below is the original two-bidder landing page evaluated at the
previous 3/10 judge head. Its “VERIFIED” labels apply only to that toy
construction and are superseded by the claim-by-claim assessment above.

### Repro — CA-AMA: Correlation-Aware Affine Maximizer Auction (TA3NDHgNJh)

Clean-room reproduction of *Enhancing Affine Maximizer Auctions with Correlation-Aware
Payment* (Sun, Xia, Chu, Deng; arXiv [2602.09455](https://arxiv.org/abs/2602.09455)), for the
[ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `TA3NDHgNJh`.

**Theorem 3.3 constructed instance** (single item, 2 bidders, perfectly negatively correlated):
v₁ ~ equal-revenue density `f(v)=ε/((1−ε)v²)` on [ε,1]; v₂=ε/(1−ε)·(1−v₁). Optimal revenue
(Crémer-McLean full surplus) `REV_F = E[v₁] = ε·ln(1/ε)/(1−ε)` (v₁≥v₂ always → bidder 1 always wins).

### Results (legacy two-bidder scope)

| Claim | Verdict | Headline evidence |
|---|---|---|
| **C1** classic AMA performs arbitrarily poorly | **VERIFIED** | classic-AMA revenue / REV_F → 0 as ε→0 (0.323→0.207→0.144, ≈1/ln(1/ε)); even the best classic AMA (optimized reserve) stays ≪ REV_F. |
| **C2** CA-AMA achieves optimal revenue | **VERIFIED** | CA-AMA with correlation-aware payment `p₁(v₂)=v₁` extracts full surplus → revenue = E[v₁] = REV_F exactly (to 1e-9); DSIC (payment depends only on v₂) + IR (u₁=0). |

6/6 pytest tests pass. Negative control: on iid bidders, full surplus (0.667) > Myerson optimal (0.417) — full-surplus extraction is impossible without correlation, confirming C2's mechanism requires correlation.

### Reproduce the legacy check
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy scipy pytest
python repro/src/run_caama.py    # C1 + C2 + controls
python -m pytest repro/tests/
```

### Legacy verification method
- **REV_F** two-method: closed form `ε·ln(1/ε)/(1−ε)` vs `scipy.integrate.quad`.
- **C1:** classic-AMA (second-price) revenue = E[v₂]; ratio E[v₂]/E[v₁] ≈ 1/ln(1/ε) → 0; best-with-reserve also → 0.
- **C2:** CA-AMA payment `p₁(v₂)=v₁` (a function of v₂ alone → DSIC); revenue = E[v₁] = REV_F.
- **Negative control:** iid U[0,1] — Myerson optimal (5/12) < full surplus (2/3), so full-surplus extraction needs correlation.

### Legacy scope and disclosures
- Single-item, 2-bidder, the Theorem 3.3 constructed instance (Sections 3 + Appendix B). The
  paper's multi-item neural-training experiments (Section 5, GPU) are out of scope.
- Official code `Haoran0301/CA-AMA` (`train_caama.py`, `auction.py`) is PyTorch/GPU training
  for the empirical section; this repro verifies the closed-form Theorem 3.3 identities directly
  (numpy/scipy). Core math is clean-room.

Logbook: https://huggingface.co/spaces/DineshAI/TA3NDHgNJh
