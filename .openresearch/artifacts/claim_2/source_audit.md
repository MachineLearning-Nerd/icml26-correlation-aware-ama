# Source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Retrieved with an explicit browser User-Agent: `2026-07-23T15:56:49Z`
- Source SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Anchor(s): `S3.Thmtheorem3; Theorems B.1 and B.2`
- Domain and quantifiers: single item; every positive integer n; all bidder-independent F for equality; every delta>0 for correlated separation
- Machine-checkable contract: Certify both inequalities giving D-CA=D-AMA on independent product supports, and audit the literal all-n correlated construction with D-CA=REV and D-AMA<delta*REV, including n=1.

## Construction notation

The theorem statement uses epsilon as the requested approximation factor, while
Appendix B uses epsilon for the equal-revenue support endpoint and delta for the
requested factor. The verifier renames the endpoint `eta`, its rival slope
`eta_1`, and the requested factor `delta`.

The Appendix-B construction requires a rival bidder to reveal bidder 1's value,
but the source statement says “any number of bidders” without an `n>=2`
qualification. This audit therefore does not silently narrow the quantifier.
For `n=1`, every DSIC/IR mechanism is a mixture of posted prices, while a
deterministic AMA implements every posted price. Thus, for every distribution
with positive optimal revenue, `REV_D-AMA=REV`; the claimed separation is
impossible for any requested factor below one.

A zero-revenue distribution makes the displayed inequality `0<=delta*0`
vacuously true. The paper's phrase “arbitrarily small fraction,” its ratio
argument, and the campaign's non-vacuity rule require positive optimal revenue;
the audit records this semantic assumption explicitly.

## Audited formula discrepancy

Appendix B defines `v_2=eta_1(1-v_1)` but prints the extracting payment as
`1-v_2/eta`. Since `eta_1<eta`, that expression is not `v_1`. The executable
construction uses the algebraically necessary inverse `1-v_2/eta_1=v_1`.
This is recorded as a proof-formula correction, not hidden as an exact match.
