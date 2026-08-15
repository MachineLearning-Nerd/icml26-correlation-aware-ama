# Release gate — canonical CA-AMA claim campaign

Status: **PUBLISHED — AWAITING LIVE JUDGE**

- Previous live judged score: `3/10`
- Conservative projected score range after the proposed change: `6–8/10`
- Best-supported possible new score: `8/10` (**forecast, not a judge result**)

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 1 | 2 | MEDIUM | FALSIFIED | Literal `n=1` counterexample satisfies the stated domain; risk is that the judge reads an implicit `n≥2` restriction |
| 2 | 1 | 2 | MEDIUM | FALSIFIED | Literal one-bidder correlated separation is impossible; intended independence and multi-bidder identities pass |
| 3 | 1 | 2 | HIGH | VERIFIED | Rival-only surcharge cancels for every own report; symbolic, exhaustive, and negative-control checks pass |
| 4 | 0 | 0–2 | MEDIUM | BLOCKED | Exact full-scale seed 1 matches both revenue values within 1.47%; Table 1's five-seed aggregate is still running |
| 5 | 0 | 0 | LOW | BLOCKED | Four distinct routes are complete; the public generator conflicts with the paper and no valid falsification was found |

Current total score: **3/10**. Conservative projected total: **6–8/10**.
Best-supported possible total: **8/10**, subject only to the live judge.
Claims 1–5 all have materially stronger evidence than at the previous judge
head. Claims 4 and 5 remain BLOCKED for the reasons above.

The approved 133-path text-only release was published to the existing Space
`DineshAI/TA3NDHgNJh` and re-downloaded at exact revision
`615386b5740671c4481b076588c796192449516a`. The final validator passes.
No second Space was created.

## Scientific release

- Exact full-scale Claim 4 experiment:
  `release/claim-4-exact-seed-evidence`
- Evidence Git SHA:
  `c78365aba5ba515c53984a3d239c9edaab272fe2`
- Run: `404b2395-c341-453e-8f0e-d7aa9b583e09`
- Runtime: 8h28m local CPU
- Result: 27 tests passed; exact revenues `3.090107` and `3.567311`
- Fixed command:

```bash
uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests
```

The exact five-seed aggregate run
`3deb95be-0518-43a4-a802-d2e19ad5c63d` remains in progress. It is disclosed
but is not part of the terminal evidence used for this release.

## Gate checks

| Gate | Result |
|---|---|
| Every claim has VERIFIED, FALSIFIED, or BLOCKED | PASS |
| Previously accepted theory regressions remain represented | PASS |
| Exact Claim 4 raw evidence and independent checker are durable | PASS |
| Negative controls reject corrupted or rival-reversed evidence | PASS |
| No toy or undertrained result is promoted | PASS |
| Judged `1c13494…` tree retained | PASS — 19/19 paths |
| Published `2a8d251…` tree retained | PASS — 120/120 paths |
| Canonical logbook structure | PASS against the required canonical slug |
| Existing-Space slug exception | DISCLOSED — `TA3NDHgNJh` cannot satisfy the new `repro-` naming rule without violating the no-second-Space requirement |
| Poster preflight, style, measure, and polish | PASS — 0 warnings |
| Final poster PDF geometry | PASS — 60×36 inches |
| Candidate logbook JSON, files, slugs, and links | PASS |
| Exact text-only upload allowlist | PASS — 133 paths |
| SHA-256 upload manifest | PASS — 133 paths |
| Manifested scientific evidence | PASS — 105 files |
| High-signal secret scan | PASS — 0 hits |

The reproducible release validator is:

```bash
uv run --frozen python release/validate_release.py \
  --protected <judged-space-1c13494...> \
  --protected <published-space-2a8d251...> \
  --candidate <candidate-space-canonical>
```

The exact upload surface is `release/hf_upload_allowlist.txt`, authenticated
by `release/hf_upload_manifest.sha256`. Every path is UTF-8 text with no NUL
bytes.

## Compute and cost

- Exact Claim 4 seed: 8h28m local CPU.
- Completed Claim 5 HF `cpu-upgrade` pilot: 1h58m.
- GPU usage: none.
- Local billed cost: $0 incremental.
- Hugging Face billed amount: unavailable from the orchestration record and
  therefore not inferred.

No score increase is claimed until a new live judge verdict exists.

The GitHub copy of the historical reader snapshot is intentionally sanitized:
private local paths are redacted and the generated `.trackio/` export is kept
outside the repository. This is a provenance mirror, not a new Hugging Face
upload or a claim that the hosted Space changed.

## Completed publication

1. Parent revision verified:
   `2a8d251ffd164a986851643d500ab774608b4b41`.
2. Canonical logbook committed at:
   `615386b5740671c4481b076588c796192449516a`.
3. Exact revision re-downloaded; both protected snapshots remain complete,
   payload hashes match, and `.gitattributes` remains byte-identical.
4. Release status marked awaiting live judge; score remains 3/10.
