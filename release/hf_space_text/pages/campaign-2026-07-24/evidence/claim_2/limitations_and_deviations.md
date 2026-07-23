# Limitations and deviations

- The Appendix-B payment denominator is corrected from `eta` to `eta_1`; the
  printed formula otherwise fails its own full-extraction identity.
- The universal independent-distribution conclusion rests on the paper's
  algebraic boost-shift proof. The independent implementation exhaustively
  checks finite product supports across five cases but is not a proof assistant.
- Allocation-favouring tie breaking against the reserve is used, matching the
  proof step that treats a weak score inequality as allocation.
- The bidder-independent equality remains supported for `n=1`; the FALSIFIED
  verdict concerns the theorem's correlated-separation half.
- As in Claim 1, the `n=1` contradiction requires the intended non-vacuous,
  positive-revenue reading of “arbitrarily poor.”
