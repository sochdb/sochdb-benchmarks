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
