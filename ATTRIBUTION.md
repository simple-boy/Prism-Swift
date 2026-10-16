# Attribution and reuse

Prism-Swift is a new synchronous multimodal storage/API implementation. It was
developed in the same collaboration as [Chronicle-Memory](https://github.com/simple-boy/Chronicle-Memory)
and the local Chronicle-Memory-Tracks service. The owner authorized reuse and MIT licensing.
The audited public repository revision was `ce888444beba2e2611335356b480b179589c8986`.

Reused ideas include source provenance, user isolation, request receipts, SQLite WAL/FULL,
time-aware evidence selection, and the earlier local dHash/color retrieval heuristic.
The old text fact extractor, its SQLite schema, per-image remote caption writer and
HTTP server were not copied into the new production package. The optional pinned
old ZIP is used only by an isolated benchmark runner, not by production.

[V-Mem](https://arxiv.org/abs/2608.01543) is a close publicly available preprint: raw evidence,
round association, modality routing, rank fusion and neighbours must not be claimed
as novel simply because they are combined here. See `docs/related_work.md`.

TinyCLIP is optional and disabled by default. Model: `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`, revision
`a2a8c6eaa2549ad66eb7c31b85022bf58273a26c`; model-card license MIT. Weights are
not included in the source archive. Transformers, PyTorch and other dependencies
retain their own licenses. Model policy/competition division remains separate
from copyright permission.

The optional production API hook performs bounded candidate reranking at Search,
with a pinned local CPU model; Add remains model-free. It was exercised locally
on synthetic data. This is distinct from the standalone encoder probe and the
experimental Add-time vector pipeline in `evaluation/compare_visual_pipeline.py`.
The optional visual Docker image has not been built or validated. Enabling the
reranker changes the frozen database configuration and requires an independent
database/volume and declared version. An Industrial submission is recommended;
Academic/Open-source eligibility requires organizer confirmation. No cloud or
official Smoke/Full success is asserted by this attribution file.

The project was renamed from its internal development name to Prism-Swift on
2026-10-16 at the owner's request. Historical measured JSON reports retain their
original source hashes and paths; those are provenance, not claims of final-source identity.
