# Conditional-support method

- Full paper setting: three bidders, ten items, alpha=0.5.
- Five deterministic mechanism-training seeds and one fixed 20,000-profile
  evaluation set.
- A 501-point reserve grid is optimized from 100,000 iid item draws per seed.
- The baseline is a valid separable reserve VCG/AMA.
- CA-AMA adds the exact rival-only conditional utility infimum. Given two rival
  values with sum `s`, the own-value support infimum is `max(0.5-s, 0)`.
- This construction is DSIC and support-wise IR by definition; all inequalities
  are also checked samplewise.
- A reversed-profile pCor control destroys the rival matching and must increase
  IR regret.
