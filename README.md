# Repro — CA-AMA: Correlation-Aware Affine Maximizer Auction (TA3NDHgNJh)

Clean-room reproduction of *Enhancing Affine Maximizer Auctions with Correlation-Aware
Payment* (Sun, Xia, Chu, Deng; arXiv [2602.09455](https://arxiv.org/abs/2602.09455)), for the
[ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `TA3NDHgNJh`.

**Theorem 3.3 constructed instance** (single item, 2 bidders, perfectly negatively correlated):
v₁ ~ equal-revenue density `f(v)=ε/((1−ε)v²)` on [ε,1]; v₂=ε/(1−ε)·(1−v₁). Optimal revenue
(Crémer-McLean full surplus) `REV_F = E[v₁] = ε·ln(1/ε)/(1−ε)` (v₁≥v₂ always → bidder 1 always wins).

## Results (all CPU, exact / numerical integration)

| Claim | Verdict | Headline evidence |
|---|---|---|
| **C1** classic AMA performs arbitrarily poorly | **VERIFIED** | classic-AMA revenue / REV_F → 0 as ε→0 (0.323→0.207→0.144, ≈1/ln(1/ε)); even the best classic AMA (optimized reserve) stays ≪ REV_F. |
| **C2** CA-AMA achieves optimal revenue | **VERIFIED** | CA-AMA with correlation-aware payment `p₁(v₂)=v₁` extracts full surplus → revenue = E[v₁] = REV_F exactly (to 1e-9); DSIC (payment depends only on v₂) + IR (u₁=0). |

6/6 pytest tests pass. Negative control: on iid bidders, full surplus (0.667) > Myerson optimal (0.417) — full-surplus extraction is impossible without correlation, confirming C2's mechanism requires correlation.

## Reproduce
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy scipy pytest
python repro/src/run_caama.py    # C1 + C2 + controls
python -m pytest repro/tests/
```

## Verification method
- **REV_F** two-method: closed form `ε·ln(1/ε)/(1−ε)` vs `scipy.integrate.quad`.
- **C1:** classic-AMA (second-price) revenue = E[v₂]; ratio E[v₂]/E[v₁] ≈ 1/ln(1/ε) → 0; best-with-reserve also → 0.
- **C2:** CA-AMA payment `p₁(v₂)=v₁` (a function of v₂ alone → DSIC); revenue = E[v₁] = REV_F.
- **Negative control:** iid U[0,1] — Myerson optimal (5/12) < full surplus (2/3), so full-surplus extraction needs correlation.

## Scope & honest disclosures
- Single-item, 2-bidder, the Theorem 3.3 constructed instance (Sections 3 + Appendix B). The
  paper's multi-item neural-training experiments (Section 5, GPU) are out of scope.
- Official code `Haoran0301/CA-AMA` (`train_caama.py`, `auction.py`) is PyTorch/GPU training
  for the empirical section; this repro verifies the closed-form Theorem 3.3 identities directly
  (numpy/scipy). Core math is clean-room.

Logbook: https://huggingface.co/spaces/DineshAI/TA3NDHgNJh
