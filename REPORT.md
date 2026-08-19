# Scoped reproduction report

## Final verdict

| Claim | Verdict | Meaning |
| --- | --- | --- |
| C1 | `FALSIFIED_LITERALLY_INTENDED_N_GE_2_SUPPORTED` | The printed “any number of bidders” wording fails at n=1; the intended n≥2 construction has high-precision support. |
| C2 | `FALSIFIED_LITERALLY_INTENDED_COMPONENTS_SUPPORTED` | Independence and multi-bidder components pass, but a one-bidder market cannot realize the stated correlated separation. |
| C3 | `VERIFIED` | Rival-only payment cancellation, finite mechanism checks, and an own-report negative control support the DSIC property. |
| C4 | `BLOCKED_EXACT_2048_CORE_UNAVAILABLE` | Seed-1 AMenuNet evidence is close and the pCor proxy passes bounded checks, but the exact released five-seed 3×10 core is unavailable. |
| C5 | `BLOCKED_SOURCE_CODE_MISMATCH_AND_UNDERTRAINED_ROUTES` | Four routes expose data-contract and training limitations without a valid falsification or paper-faithful full optimization. |

Overall status is `INCONCLUSIVE_SCOPED_FINDINGS_WITH_BLOCKED_PAPER_LEVEL_EMPIRICAL_CLAIMS`;
`publication_allowed` is `false` for a complete empirical reproduction or score. The historical
3/10 judge result is provenance only.

## Claim production and evidence boundary

Theory claims C1–C3 are produced by `repro/src/theory_campaign.py` and checked by the durable
verifier outputs under `.openresearch/artifacts/claim_1/` through `claim_3/`. C4 records source,
conditional-support, pCor, and exact seed-1 routes; the missing released 2048-menu core prevents
promotion. C5 records source/data-contract, paper-semantics, direct mechanism, and falsification
routes; undertrained or mismatched routes are not promoted to either success or falsification.

The authoritative human ledger is [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md), the normalized records
are [claims.json](claims.json) and [reproduction_verdicts.json](reproduction_verdicts.json), and
the evidence surface is hash-bound in [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json).

## Branch and publication policy

`main` is the reader-facing publication surface. The 22 retained `audit/*`, `baseline/*`,
`experiment/*`, and `release/*` branches preserve distinct routes and are mapped in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md). The root [verify_final.py](verify_final.py) checks all 23
public branches, canonical MachineLearning-Nerd attribution, source pins, claim boundaries,
evidence hashes, and the existing lightweight publication gate without launching training.

Thank you to Haoran Sun, Xuanzhi Xia, Xu Chu, and Xiaotie Deng for releasing the CA-AMA paper and
official implementation for independent study; see [AUTHOR_THANK_YOU.md](AUTHOR_THANK_YOU.md).
