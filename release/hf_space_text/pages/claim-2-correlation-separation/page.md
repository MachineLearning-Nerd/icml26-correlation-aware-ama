# Claim 2: independence equality and correlated separation

---
<!-- trackio-cell
{"type":"markdown","id":"cell_claim_2_20260724","created_at":"2026-07-24T00:02:00+00:00","title":"Claim 2 evidence"}
-->
**Paper statement.** Under bidder-independent distributions, CA-AMA and
classic AMA have equal revenue; under correlation, CA-AMA can attain the
optimal revenue while classic AMA is arbitrarily poor.

**Observed evidence.** The independence equality follows exactly because the
conditional rival-only surcharge cannot exploit bidder information when
valuations are independent. On the correlated construction, CA-AMA equals
full-surplus revenue to at most `1e-14` over the epsilon sweep while the
classic ratio tends to zero. The literal one-bidder scope cannot exhibit a
correlation-aware separation because there is no rival profile.

**Assessment: FALSIFIED under the literal all-bidder scope; the independence
identity and intended multi-bidder correlated separation are supported.**
The remaining risk is whether the theorem implicitly excludes `n=1`.

Negative control: for iid uniform bidders, full surplus `2/3` exceeds
Myerson-optimal revenue `5/12`; correlation-free full-surplus extraction is
correctly rejected.
