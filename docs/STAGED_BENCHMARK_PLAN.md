# Staged Benchmark Plan

## Why staged

The current benchmark server does not have enough free root-disk capacity for a
clean, honest `1TB` run today.

So we should run staged sizes:

1. `10GB`
2. `100GB`
3. `250GB`
4. `1TB` only after storage is expanded

## Current benchmark workspace

Server:

- `/root/sochdb-benchmark-runs/datasets`
- `/root/sochdb-benchmark-runs/results`
- `/root/sochdb-benchmark-runs/logs`
- `/root/sochdb-benchmark-runs/work`

## Pilot runner

The first reusable runner is:

- `scripts/run_sochdb_grpc_pilot.sh`

It wraps the gRPC retrieval benchmark and writes:

- result JSON
- metadata JSON
- full log

## Example

```bash
DATASET_DIR=/root/sochdb-benchmark-runs/datasets/scifact \
EMBEDDING_DIR=/root/sochdb-benchmark-runs/datasets/scifact-embeddings \
/root/sochdb-benchmark-runs/work/run_sochdb_grpc_pilot.sh
```

## Quality-first benchmark lane

For quality work, we should avoid mixing too many variables at once.

The evaluation order should be:

1. fix the dataset
2. fix the embedding model
3. sweep HNSW settings
4. compare recall / nDCG / latency
5. only then compare different embedding models

The first quality sweep runner is:

- `scripts/run_sochdb_grpc_quality_sweep.sh`

It compares three useful HNSW profiles:

- `fast`
- `balanced`
- `quality`

and writes:

- per-run result JSON
- summary JSON
- summary table text

## Server-only embedding bakeoff

If we want to improve retrieval quality, the next likely lever after HNSW sweeps is embeddings.

The server-only runner for that is:

- `scripts/run_sochdb_embedding_bakeoff.sh`

It does this on the server:

1. generates embeddings for each configured model
2. runs the same HNSW quality sweep for each embedding set
3. writes one result directory per model

That keeps the heavy work off the laptop and keeps the comparison methodology clean.
