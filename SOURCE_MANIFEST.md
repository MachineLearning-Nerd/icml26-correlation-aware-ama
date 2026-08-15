# Source manifest

## Paper

- Title: *Enhancing Affine Maximizer Auctions with Correlation-Aware Payment*.
- Authors: Haoran Sun, Xuanzhi Xia, Xu Chu, and Xiaotie Deng.
- arXiv: [2602.09455](https://arxiv.org/abs/2602.09455).
- HTML source: [arxiv.org/html/2602.09455](https://arxiv.org/html/2602.09455).
- OpenReview challenge identifier: `TA3NDHgNJh` ([record link](https://openreview.net/forum?id=TA3NDHgNJh)).
- Source boundary: the claim ledger uses the paper's printed theorem
  statements, assumptions, algorithms, distributions, and reported numbers.
  A reproduction result does not imply author endorsement.
- Retrieval record: `sources.json`, recorded 2026-08-15.

## Official implementation

- Repository: [Haoran0301/CA-AMA](https://github.com/Haoran0301/CA-AMA).
- Pinned upstream commit: `ed2af19ed02c70b58efdf705635981241222d045`.
- Local snapshot: [`upstream/`](upstream/).
- Relationship: copied reference source, not a submodule and not an author
  release made by this repository.

## Audit evidence

- Clean-room executable code: [`repro/`](repro/).
- Durable claim artifacts: [`.openresearch/artifacts/`](.openresearch/artifacts/).
- Technical report: [`reports/caama-claim-campaign/report.md`](reports/caama-claim-campaign/report.md).
- Historical reader snapshot: [`release/hf_space_text/`](release/hf_space_text/).
- Historical campaign evidence SHA: `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`.
- Historical fixed command:
  `uv run --frozen python repro/src/run_caama.py && uv run --frozen python -m pytest -q repro/tests`.

## Source/code discrepancy recorded by the audit

The paper's Linear Mixture Asymmetric description uses a Bernoulli mixture,
while the public `generate_data_22` implementation uses convex interpolation.
That difference is recorded as an unresolved data contract, not silently
collapsed into a single reproduction protocol.

## Citation

```bibtex
@article{sun2026enhancing,
  title={Enhancing Affine Maximizer Auctions with Correlation-Aware Payment},
  author={Sun, Haoran and Xia, Xuanzhi and Chu, Xu and Deng, Xiaotie},
  journal={arXiv preprint arXiv:2602.09455},
  year={2026},
  url={https://arxiv.org/abs/2602.09455}
}
```

## Thank-you note

Thank you to Haoran Sun, Xuanzhi Xia, Xu Chu, and Xiaotie Deng for making the
CA-AMA paper and official implementation available for independent study.
This repository is an independent reproducibility audit and does not
represent the authors' implementation, approval, or endorsement.
