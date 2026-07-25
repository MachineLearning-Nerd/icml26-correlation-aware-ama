# Limitations and deviations

- The verifier establishes the deterministic-AMA part scored by Claim 1. It
  source-audits, but does not assign a separate verdict to, Proposition 3.1's
  additional finite-menu randomized-AMA strict-gap clause.
- The literal `n=1` quantifier is not excluded. It falsifies the intended
  positive-revenue claim; restricting the theorem to `n>=2` repairs it.
- A zero-revenue distribution satisfies the displayed inequality vacuously.
  This audit follows the paper's stated “fraction” interpretation and the
  campaign rule against vacuous checks by requiring positive optimal revenue.
- The certificate validates the paper's uniform upper-bound proof; it does not
  numerically optimize every possible AMA parameterization.
