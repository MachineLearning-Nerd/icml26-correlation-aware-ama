# Limitations and deviations

- This route completes only one of the five paper training seeds and therefore
  remains BLOCKED even if its numeric outcome aligns exactly.
- Menu size 2,048 and initial gamma 8 are inferred from the transposed 10x3
  Table-3/released-script entry; the paper never supplies a 3x10 row.
- Constant-context vectorization is algebraically regression-tested against the
  released implementation but is a CPU execution optimization.
- A divergent non-convex training result would not by itself falsify Table 1.
