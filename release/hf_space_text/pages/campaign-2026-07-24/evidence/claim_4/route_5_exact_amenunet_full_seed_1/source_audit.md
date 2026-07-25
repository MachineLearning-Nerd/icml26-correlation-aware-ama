# Claim 4 full-seed source audit

- Paper: `https://ar5iv.labs.arxiv.org/html/2602.09455`
- Paper SHA-256: `2018a533559d5777eedfb1a0bb043bd490b07b2d89b8f5d3edf3adad4ad72e4f`
- Released code: `ed2af19ed02c70b58efdf705635981241222d045`
- Table 1 target: Randomized AMA `3.1363`, CA-AMA `3.6205`, IR regret
  `0.0031`, ex-post-IR revenue `3.5623`, averaged over five seeds.
- Section 5: 32,000 iterations, 1,024 batch size for larger settings,
  temperature 500, fixed 20,000-profile test set.
- Algorithm 1: exact argmax AMA payments in post-training.
- Unresolved source discrepancy: Table 1 says `3x10`; Table 3 and the released
  shell matrix provide only a transposed `10x3` entry. The 2,048-menu and
  initial-gamma-8 choices are therefore disclosed inferences.
