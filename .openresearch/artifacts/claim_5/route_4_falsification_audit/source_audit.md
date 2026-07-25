# Claim 5 fourth-route source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Retrieved: `2026-07-23T15:56:49Z`
- SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1.p3`
- Training anchor: `S5.p2`

The exact Table 1 row gives `1.7135`, `1.9359`, `0.0052`, and `1.8553`.
The prose distribution is a Bernoulli mixture, not the convex interpolation
implemented by the released `generate_data_22` function. This audit uses the
paper's literal Bernoulli statement.
