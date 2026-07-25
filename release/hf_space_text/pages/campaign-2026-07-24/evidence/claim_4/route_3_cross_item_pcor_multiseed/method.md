# Cross-item pCor multi-seed method

- Fixed reserve-AMA cores come from the full-scale Route 2 optimization.
- A distinct three-layer MLP for each bidder consumes only the other bidders'
  20 values, exactly preserving DSIC.
- The pCor learner optimizes revenue plus adaptive average IR-regret penalty.
- Five seeds, 2,000 updates per seed, and batch size 1,024.
- A held-out 10,000-profile validation set selects the largest global payment
  scale with mean IR regret no greater than the paper's 0.0031. Scaling a
  rival-only payment preserves its rival-only dependence and DSIC.
- A disjoint fixed 20,000-profile hard test is used for each training seed.
- Student-t 95% intervals use training seeds as the independent units.
- Negative pCor outputs are clipped to zero at evaluation, as in the release.
- Reversing rival profiles is the negative control.
