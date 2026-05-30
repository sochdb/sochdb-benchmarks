# Staged Benchmark Plan

Start with [`SERVER_BENCHMARK_STATUS.md`](./SERVER_BENCHMARK_STATUS.md) for the
current published results.

## Why staged

The current benchmark server does not yet have the storage profile for a clean
final `1TB` benchmark claim.

So the scale path stays staged:

1. `10GB`
2. `100GB`
3. `250GB`
4. `1TB` after storage expansion

## Current `10GB` status

The first staged dataset is:

- dataset: `synthetic_10gib_768d`
- vectors: `3,495,253`
- dimension: `768`

Published result summary:

- build worked and produced about a `10,069 MB` index
- corrected steady-state search result is about `506.63 QPS`
- corrected sequential mean latency is about `1.97 ms`
- one-time index load is about `106.85 s`

Important note:

- the original slow search number from the bulk CLI harness was a methodology
  artifact, not the final engine result

## Reusable scripts

- `scripts/generate_staged_vector_dataset.py`
- `scripts/run_sochdb_stage_vector.sh`
- `benchmarks/run_bulk_vector_workload.py`

## What this lane is for

This lane measures:

- build throughput
- search QPS
- search latency
- index size at scale

This lane does not measure:

- retrieval relevance quality
- semantic usefulness on real-world corpora

## Next step

Use the corrected native steady-state methodology for the next stages:

1. publish `100GB`
2. publish `250GB`
3. defer `1TB` until storage is expanded
