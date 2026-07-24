# Claim 4 exact AMenuNet full-seed method

- Seed 1; literal Dirichlet Value Share(alpha=0.5), 3 bidders x 10 items.
- Released dropout-free transformer AMenuNet parameterization with 2,048 menus.
- Randomized AMA: 32,000 optimizer updates.
- CA-AMA: 16,000 mutual + 16,000 pCor-only post updates.
- Batch 1,024; softmax temperature 500; fixed seed-2002 20,000-profile test.
- Paper-stated three-linear-layer rival-only ReLU pCor MLP.
- Exact hard-argmax AMA outcome/payment in post-training and evaluation.
- Rival-profile reversal and zero-pCor identity are negative controls.
