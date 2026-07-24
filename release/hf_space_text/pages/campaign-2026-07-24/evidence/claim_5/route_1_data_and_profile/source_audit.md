# Source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Anchors: Table 1 `S4.T1`; implementation `S5.p3`; distribution `S5.SS1`
- Exact source statement: Table 1: Linear Mixture alpha=0.6, 2x5 Asym, Randomized AMA 1.7135, CA-AMA 1.9359 (IR regret 0.0052), ex-post-IR CA-AMA 1.8553.
- Claim contract: Using 2 bidders, 5 items and the literal Bernoulli(alpha=0.6) asymmetric mixture, reproduce five-seed test revenue, uncertainty, and empirical IR regret for equally sized mechanisms.
- Official code revision audited: `ed2af19ed02c70b58efdf705635981241222d045`

This profiling node does not issue a claim verdict. It validates the input
distribution contract and measures feasibility before full training.
