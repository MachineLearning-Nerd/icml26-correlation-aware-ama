# Limitations and deviations

- Menu size 2,048 and initial gamma 8 are inferred from the conflicting
  transposed `10 x 3` row; no released `3 x 10` checkpoint or command exists.
- "32,000 iterations" is interpreted as optimizer updates. Released scripts
  use outer data-generation loops containing multiple minibatch updates, so
  the paper and public code do not expose one unambiguous counter.
- The paper states a three-layer ReLU pCor network. Released mutual training
  instead uses max-minus-max, while released post-training uses ReLU.
- Constant-context vectorization is algebraically regression-tested against
  released AMenuNet but is a CPU execution optimization.
- Concurrent workers affect wall-clock time, not seeds, samples, or contracts.
- A divergent non-convex optimization outcome is BLOCKED, not FALSIFIED.
