# Reproducible developer experiments

All inputs are generated locally. No official evaluation records, gold answers or
private images are included. None of these scripts measures the official Answer/Eval score.

## Current implementation

```bash
PYTHONPATH=src python scripts/benchmark.py --mode all --cores 2 --ablations \
  --output results/my-benchmark.json
PYTHONPATH=src python scripts/benchmark.py --mode performance --strategies legacy_none \
  --output results/my-legacy-benchmark.json
PYTHONPATH=src python scripts/scale_benchmark.py --messages 10000 \
  --output results/my-scale.json
```

The legacy mode defaults to the seven bundled, hash-verified Python source files in
`baselines/legacy_tracks`. They came from the owner's earlier October 14 diagnostic
release, whose archive SHA256 is recorded in its `SOURCE_MANIFEST.json`. The adapter
disables all models and adds safe aggregate metrics. It is a measured no-model lower
cost baseline, not a reproduction of the remote Quanzil system or the official slow run.
The baseline files are historical comparison sources, not part of the serving image.

For real CPU visual experiments first install the pinned optional dependencies and
download/verify the public weights as described in `docs/visual_reranker.md`. Then:

```bash
python evaluation/compare_visual_pipeline.py --model-dir models/tinyclip \
  --output results/my-visual-write-comparison.json
python evaluation/compare_query_reranker.py --model-dir models/tinyclip \
  --output results/my-query-reranker.json
```

These scripts use actual fixed TinyCLIP inference, not a fake vector lookup. Outputs
include per-source fingerprints, generated seeds and per-query metrics. Query-only
tests use actual primary retrieval candidates, without gold filtering. Images are
easy generated shapes; results cannot establish natural-image or official accuracy.

## Exact October 15 A/B/C source snapshot

The earlier A/B/C report `docs/visual_pipeline_comparison.json` predates the project
rename, the persistent-WAL-connection optimization and later serving fixes. Its exact
seven source files and hashes are preserved under `frozen_20261015` so that the report
does not falsely refer to today's modified code. That folder retains its historical
Python namespace solely to preserve the measured source hashes; it is not imported
by the current service. No old database, memory or model weights are included.

```bash
python evaluation/frozen_20261015/evaluation/compare_visual_pipeline.py \
  --model-dir "$(pwd)/models/tinyclip" \
  --output "$(pwd)/results/my-frozen-visual-comparison.json"
```

Use the original report's Python/library versions and platform for a close comparison.
Even with identical source and seeds, CPU/disk scheduling can change latency. Do not
rewrite the checked-in historical result files when rerunning; use new output names.
