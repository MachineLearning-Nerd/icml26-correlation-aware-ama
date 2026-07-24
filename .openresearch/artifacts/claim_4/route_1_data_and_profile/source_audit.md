# Source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Anchors: Table 1 `S4.T1`; implementation `S5.p3`; distribution `S5.SS1`
- Exact source statement: Table 1: Dirichlet Value Share alpha=0.5, 3x10, Randomized AMA 3.1363, CA-AMA 3.6205 (IR regret 0.0031), ex-post-IR CA-AMA 3.5623.
- Claim contract: Using 3 bidders, 10 items, T_j~U[0.5,1], and w_j~Dirichlet(0.5,0.5,0.5), reproduce five-seed test revenue and uncertainty for equally sized Randomized AMA and CA-AMA.
- Official code revision audited: `ed2af19ed02c70b58efdf705635981241222d045`

This profiling node does not issue a claim verdict. It validates the input
distribution contract and measures feasibility before full training.
