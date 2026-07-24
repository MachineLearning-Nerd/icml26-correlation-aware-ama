# Method

The correlated part reuses the all-n construction and checks pointwise that
`pCor_1(V_-1)=1-v_2/eta_1=v_1`, so CA-AMA extracts the full surplus. The
independent part implements Appendix B's boost-shift transform on Cartesian
finite supports and enumerates every profile, checking pointwise that the
transformed AMA payment dominates the feasible CA-AMA payment. Since CA-AMA
contains AMA at `pCor=0`, the two optimal revenues are equal.

For the correlated-separation half, apply the one-bidder posted-price
representation to the literal `n=1` scope. It proves that a positive-revenue
separation cannot hold for every bidder count.
