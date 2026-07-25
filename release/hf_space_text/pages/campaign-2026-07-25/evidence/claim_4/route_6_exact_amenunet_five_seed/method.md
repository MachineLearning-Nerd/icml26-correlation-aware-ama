# Claim 4 method

- Five deterministic training seeds: `1,2,3,4,5`.
- Literal Dirichlet Value Share(alpha=0.5), 3 bidders x 10 items.
- Released dropout-free AMenuNet transformer parameterization.
- Menu size 2,048; initial gamma 8; gamma update 0.01; cap 20.
- Randomized AMA: 32,000 optimizer updates.
- CA-AMA: 16,000 mutual + 16,000 pCor-only post updates.
- Batch 1,024; softmax temperature 500; allocation temperature 10.
- Fixed generator-seed-2002 test set with 20,000 profiles per seed.
- Paper-stated three-linear-layer, rival-only ReLU pCor network.
- Exact hard-argmax AMA outcomes/payments in post-training and evaluation.
- 5 isolated spawn workers, each restricted to
  2 PyTorch CPU threads.
- Uncertainty is a two-sided 95% Student-t interval over five seed means.
- Independent verification recomputes all metrics from 100,000 raw rows.
