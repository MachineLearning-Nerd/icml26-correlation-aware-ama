# Claim 4: Dirichlet Value Share Table 1

---
<!-- trackio-cell
{"type":"markdown","id":"cell_claim_4_20260724","created_at":"2026-07-24T00:04:00+00:00","title":"Claim 4 evidence"}
-->
**Paper result.** Dirichlet Value Share, alpha `0.5`, 3 bidders × 10 items,
five seeds: randomized AMA `3.1363`, CA-AMA `3.6205`, IR regret `0.0031`,
and ex-post-IR revenue `3.5623`.

**Faithful seed-1 result.** The released AMenuNet parameterization was trained
for 32,000 baseline, 16,000 mutual-payment, and 16,000 hard-argmax
post-training updates with batch size 1,024 and a fixed 20,000-profile test
set:

| Metric | Paper | Seed 1 |
| --- | ---: | ---: |
| Randomized AMA revenue | 3.1363 | 3.090107 |
| CA-AMA revenue | 3.6205 | 3.567311 |
| IR regret | 0.0031 | 0.006133 |
| Ex-post-IR revenue | 3.5623 | 3.466351 |

All revenues are within 2.7%. Regret's absolute error is `0.0030329`, narrowly
outside the pre-registered `0.003` tolerance. The route remains **BLOCKED**
because Table 1 averages five seeds. Exact run
`404b2395-c341-453e-8f0e-d7aa9b583e09` took 8h28m on local CPU; the
pre-registered five-seed aggregate run
`3deb95be-0518-43a4-a802-d2e19ad5c63d` is in progress.

Zeroing the correlation payment reduces revenue to `3.012182`; reversing
rival profiles increases IR regret to `0.306815`. Final status will be updated
only from the five-seed independent checker and fail-closed verifier.
