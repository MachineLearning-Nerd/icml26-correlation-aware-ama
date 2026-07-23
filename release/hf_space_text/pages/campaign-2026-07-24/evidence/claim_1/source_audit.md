# Source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Retrieved with an explicit browser User-Agent: `2026-07-23T15:56:49Z`
- Source SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Anchor(s): `S3.Thmtheorem1; proof A2.16.p1--A2.22.p6`
- Domain and quantifiers: single item; every positive integer n (literal source wording: "any number of bidders"); every target delta>0; there exists a correlated distribution F with a non-vacuous positive optimal revenue
- Machine-checkable contract: Audit every positive integer n. For n>=2, construct F explicitly and certify REV_F^D-AMA < delta*REV_F. For n=1, test whether a positive-revenue separation can exist under DSIC and IR.

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
