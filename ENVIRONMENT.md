# Environment and execution boundary

- Python: 3.12, with dependencies pinned by `uv.lock`.
- Official source: `Haoran0301/CA-AMA` at the commit recorded in `sources.json`.
- Theory evidence: exact/symbolic and finite CPU checks.
- Empirical evidence: seed-1, pCor, data-contract, and CPU-feasibility routes; the exact released five-seed neural core is unavailable.
- Final verification: `python3 verify_final.py` checks committed artifacts and the lightweight publication gate without launching training.

The historical campaign may require substantial CPU time. The final dossier check is not a rerun of
the neural training routes.
