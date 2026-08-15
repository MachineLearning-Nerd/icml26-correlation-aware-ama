# Reproduction status

Overall status: **INCONCLUSIVE — VERIFIED SCOPED FINDINGS WITH BLOCKED
PAPER-LEVEL EMPIRICAL CLAIMS**.

| Claim | Status | How the result is produced |
| --- | --- | --- |
| C1 — deterministic AMA can be arbitrarily poor | `FALSIFIED_LITERALLY` | `repro/src/theory_campaign.py::run_claim_1` checks the printed quantifier, produces (n\ge2) certificates, and records the exact (n=1) counterexample. |
| C2 — independence equality and correlated separation | `FALSIFIED_LITERALLY` | `repro/src/theory_campaign.py::run_claim_2` checks the intended identities and the one-bidder obstruction. |
| C3 — rival-only payment preserves DSIC | `VERIFIED` | `repro/src/theory_campaign.py::run_claim_3` checks symbolic cancellation, finite feasible domains, and an own-report negative control. |
| C4 — Dirichlet Value Share experiment | `BLOCKED` | Exact seed-1 AMenuNet evidence is close to the paper; the exact released five-seed 3 × 10 core is unavailable, so the pCor proxy is not promoted. |
| C5 — Linear Mixture Asymmetric experiment | `BLOCKED` | Four routes audit the source mismatch, training pilots, direct mechanism, and exact welfare/IR bounds; none supplies a valid paper-level verdict. |

## Evidence boundary

The historical campaign produced terminal statuses at
`.openresearch/artifacts/campaign_verdicts.json` at evidence SHA
`bf4cc9371feea65edf71ad1dc998ed88de23b7a7`. Those artifacts are preserved,
but a historical test count is not represented as a fresh run on every machine.
The current Python environment has no repository `.venv`; use Python 3.12 and
the frozen lockfile before attempting the expensive scientific command.

The previous live judge score recorded in the repository is `3/10`. No new
score, exact five-seed completion, or author endorsement is claimed.

## Lightweight publication gate

```bash
uv run --frozen python repro/src/verify_publication.py
uv run --frozen python repro/src/publication_gate.py --skip-producers
```

These checks validate documentation, evidence boundaries, branch naming, and
publication hygiene without launching training.
