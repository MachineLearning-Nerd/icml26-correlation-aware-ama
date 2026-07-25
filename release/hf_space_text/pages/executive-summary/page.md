# Executive summary

---
<!-- trackio-cell
{"type":"markdown","id":"cell_exec_summary_20260724","created_at":"2026-07-24T00:00:00+00:00","title":"Executive summary","pinned":true,"pinned_at":"2026-07-24T00:00:01+00:00"}
-->
**Three theoretical claims are resolved, and the first faithful full-scale
Table 1 run closely matches the reported revenues.** Claims 1 and 2 are
falsified only under their literal one-bidder quantifier while their intended
multi-bidder constructions pass; Claim 3 is verified by an exact rival-only
payment cancellation argument. On Dirichlet Value Share at 3 bidders × 10
items, the exact seed-1 run produced randomized-AMA revenue `3.090107` and
CA-AMA revenue `3.567311`, both 1.47% below the paper; the five-seed aggregate
is still running, so Claim 4 remains BLOCKED rather than being promoted.
Claim 5 remains BLOCKED after four distinct routes. These are reproduction
verdicts, not a new live-judge score.

## Scope & cost

|  | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Five judge claims; exact Claims 1–3; full 3×10 Claim 4 seed; four-route Claim 5 audit | All Table 1 rows at five seeds and every training interpretation |
| Hardware | Local CPU; Hugging Face `cpu-upgrade` for one Claim 5 pilot; no GPU | Paper hardware is not fully specified |
| Compute time | Claim 4 exact seed: 8h28m; Claim 5 HF pilot: 1h58m | Multiple five-seed neural experiments |
| Cost | Local CPU: $0 billed; HF billed amount unavailable | Not reported |
| Outcome | C1 FALSIFIED; C2 FALSIFIED; C3 VERIFIED; C4/C5 BLOCKED | Exact empirical adjudication still needs the terminal five-seed run |

---
<!-- trackio-cell
{"type":"figure","id":"cell_reproduction_poster_20260724","created_at":"2026-07-24T00:00:02+00:00","title":"Reproduction poster (poster_embed.html)","pinned":true,"pinned_at":"2026-07-24T00:00:03+00:00"}
-->
<!-- poster_embed.html -->
<iframe src="poster_embed.html" title="CA-AMA five-claim reproduction poster" width="100%" height="820" loading="lazy"></iframe>
