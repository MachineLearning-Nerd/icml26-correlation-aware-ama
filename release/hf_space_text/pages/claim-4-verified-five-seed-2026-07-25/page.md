# Claim 4: Dirichlet Value Share Table 1

![Five exact released-AMenuNet seeds](../campaign-2026-07-25/images/claim4_seed_stability.svg)

---
<!-- trackio-cell
{"type":"markdown","id":"cell_claim_4_20260725","created_at":"2026-07-25T00:01:00+00:00","title":"Claim 4 exact five-seed evidence"}
-->
**Verdict: VERIFIED (HIGH confidence).**

The exact claim is the five-seed Dirichlet Value Share experiment with
`alpha=0.5`, 3 bidders, 10 additive items, and the released 2,048-menu
AMenuNet parameterization.

| Metric | Paper | Observed five-seed mean | Error |
| --- | ---: | ---: | ---: |
| Randomized AMA revenue | 3.1363 | 3.085011 | −1.64% |
| CA-AMA revenue | 3.6205 | 3.572630 | −1.32% |
| IR regret | 0.0031 | 0.005835 | +0.002735 absolute |
| Ex-post-IR revenue | 3.5623 | 3.473243 | −2.50% |

Each seed used 32,000 baseline, 16,000 mutual-payment, and 16,000 hard-argmax
post-training updates, batch size 1,024, and a fixed 20,000-profile test set.
The CA-AMA mean has seed-level 95% CI `[3.563988, 3.581272]`. The paired
CA-minus-baseline improvement is `0.487619`, with 95% CI
`[0.479745, 0.495493]`.

The independent checker re-read all 100,000 raw rows. Zeroing the correlation
payment reduced revenue by `0.544717` on average, with 95% CI
`[0.527722, 0.561711]`. Reversing rival profiles increased regret by
`0.297420`, with 95% CI `[0.290231, 0.304609]`. The independent checker,
negative-control verifier, and fail-closed claim verifier all exited zero.

Exact run: `3deb95be-0518-43a4-a802-d2e19ad5c63d`. Evidence code:
`8b3aa42e97f01f1202a7bed4b38394c6be88e6f0`. Runtime: 14h21m local CPU.
No GPU was used.

Limitations remain disclosed: the public 10×3 command conflicts with the
paper's 3×10 table; menu size 2,048 is inferred from that public row; and the
paper/public-code descriptions of the payment network differ. These are
source ambiguities, but the exact pre-registered reproduction contract passes.
