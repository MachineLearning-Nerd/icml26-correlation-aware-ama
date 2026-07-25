# Claim 4 source audit

- Paper: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Retrieved: `2026-07-23T15:56:49Z`
- Paper SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Released-code commit: `ed2af19ed02c70b58efdf705635981241222d045`
- Table anchor: `S4.T1`
- Distribution anchor: `S5.SS1`
- Implementation anchors: `S5.p3`, `A1.SS4`

The source states `T_j ~ U[0.5,1]`, symmetric
`Dirichlet(alpha,...,alpha)` shares, and `v_ij = w_ij T_j`. Table 1
reports randomized AMA `3.1363`, CA-AMA `3.6205`, IR regret `0.0031`,
and ex-post-IR revenue `3.5623` for alpha `0.5`, `3 x 10`, averaged
over five seeds.

Section 5 specifies 32,000 total iterations per mechanism, batch 1,024
for larger settings, mutual/post balance, softmax temperature 500, and
a fixed 20,000-profile test set. Algorithm 1 requires exact argmax AMA
payments and utilities in post-training.

Table 3 and the released shell matrix omit `3 x 10` and instead list a
transposed `10 x 3` setting. Menu size 2,048 and initial gamma 8 are
therefore explicit inferences, not paper-stated `3 x 10` facts.
