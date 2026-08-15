# Branch audit

This audit records the experiment history before the repository cleanup. The
former `orx/` namespace came from the experiment runner. `master` becomes the
reader-facing `main` branch, and every retained experiment receives a
descriptive namespace. Branch names are navigation aids; claim verdicts come
from the durable evidence files, not from a branch label.

| Former branch | Clean branch | Tip before cleanup | Role and evidence boundary |
| --- | --- | --- | --- |
| `master` | `main` | `f763b6b` | Publication README, report, release metadata, and claim evidence. |
| `orx/canonical-five-claim-logbook-release-candidate` | `release/canonical-logbook` | `f763b6b` | Canonical five-claim logbook candidate. |
| `orx/claim-4-conditional-support-full-scale-mechanism` | `experiment/claim-4-conditional-support` | `e38e8d7` | Conditional-support full-scale mechanism route; not the exact released neural procedure. |
| `orx/claim-4-cross-item-pcor-neural-pilot` | `experiment/claim-4-cross-item-pcor` | `db55dc0` | Cross-item pCor neural pilot. |
| `orx/claim-4-exact-amenunet-cpu-feasibility` | `experiment/claim-4-amenunet-cpu-feasibility` | `7679d5b` | Exact AMenuNet CPU feasibility route. |
| `orx/claim-4-exact-amenunet-full-seed-1` | `experiment/claim-4-amenunet-seed-1` | `ac097ff` | Full-scale exact AMenuNet seed-1 training route. |
| `orx/claim-4-exact-five-seed-parallel-verification` | `experiment/claim-4-five-seed-aggregate` | `8b3aa42` | Pre-registered exact five-seed aggregate route; no incomplete result is promoted. |
| `orx/claim-4-exact-full-seed-evidence-only-gate` | `release/claim-4-exact-seed-evidence` | `c78365a` | Exact seed-1 evidence-only release gate. |
| `orx/claim-4-exact-full-seed-fixed-test-rng` | `experiment/claim-4-fixed-test-rng` | `54240a5` | Generator-scoped fixed test-RNG correction. |
| `orx/claim-4-neural-multi-seed-validation` | `experiment/claim-4-neural-multiseed` | `20a5799` | Five-seed three-layer pCor proxy validation. |
| `orx/claim-4-verified-cumulative-release-update` | `release/claim-4-cumulative-update` | `e832e58` | Cumulative Claim 4 release update. |
| `orx/claim-5-mandatory-falsification-audit` | `audit/claim-5-falsification` | `f7498f6` | Mandatory exact Claim 5 falsification route. |
| `orx/cumulative-evidence-parser-fix` | `audit/cumulative-evidence-parser-fix` | `bf4cc93` | Cumulative evidence parser regression and winning historical evidence SHA. |
| `orx/cumulative-release-candidate-evidence` | `release/cumulative-evidence` | `6e0a917` | Cumulative claim evidence bundle. |
| `orx/direct-mechanism-space-claim-5-reproduction` | `experiment/claim-5-direct-mechanism` | `9748cdf` | Claim 5 direct mechanism-space CPU-upgrade pilot. |
| `orx/frozen-judged-reproduction-baseline` | `baseline/frozen-judged-reproduction` | `0bf3ab2` | Frozen historical judged baseline and locked environment. |
| `orx/full-claim-5-paper-semantics-reproduction` | `experiment/claim-5-full-paper-semantics` | `092a1da` | Full Claim 5 paper-semantics route. |
| `orx/literal-n-1-theorem-scope-audit` | `audit/literal-theorem-scope` | `9ca2edd` | Literal one-bidder theorem-scope audit. |
| `orx/paper-faithful-data-contracts-and-cpu-profile` | `audit/data-contract-cpu-profile` | `589ef07` | Empirical distribution and CPU feasibility audit. |
| `orx/portable-cumulative-evidence-manifest` | `release/portable-evidence` | `6c9816b` | Portable cumulative evidence manifest. |
| `orx/presentation-release-candidate` | `release/presentation-candidate` | `67dac34` | Presentation/readme release candidate. |
| `orx/quantified-theory-contracts` | `audit/quantified-theory` | `747d410` | Quantified theory contracts. |
| `orx/vectorized-paper-semantics-training-pilot` | `experiment/claim-5-vectorized-paper-semantics` | `ab790eb` | Vectorized Claim 5 paper-semantics training pilot. |

## Cleanup policy

- `main` is the only default publication branch.
- No retained experiment branch is named `orx/*`; every name states its
  scientific role.
- The old names and pre-cleanup tips remain documented here for provenance.
- Historical `.trackio/` exports and private local paths are removed from the
  published branch surface; the recoverable local archive is outside Git.
- A branch tip does not override the terminal claim ledger in
  `.openresearch/artifacts/campaign_verdicts.json`.
