# Method

1. Choose `eta=exp(-2/delta)` and `eta_1=eta^2`.
2. Use Appendix B's uniform deterministic-AMA revenue upper bound
   `eta/(1-eta)+eta_1`.
3. Divide by `REV=eta*log(1/eta)/(1-eta)` and simplify independently to
   `delta/2 + eta(1-eta)delta/2 < delta`.
4. Recompute at 100-digit Decimal precision for 30 `(delta,n)` pairs.
5. Require a deliberately weak parameter choice to fail.
6. Audit `n=1` with the single-agent DSIC payment identity: every monotone
   allocation is a mixture of posted-price thresholds, so its expected revenue
   cannot exceed the best posted price, which deterministic AMA implements.
