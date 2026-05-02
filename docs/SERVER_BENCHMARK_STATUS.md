# Server Benchmark Status

This document captures the current state of the heavy benchmark lane that runs on
the hosted SochDB server instead of on a laptop.

## Why the server lane exists

Heavy benchmark work should happen on the benchmark server, not on a developer
laptop. That is especially true for:

- retrieval-quality sweeps
- embedding bakeoffs
- staged large-dataset runs
- repeatable gRPC benchmark runs against the hosted demo endpoint

Current server target:

- host: private benchmark server
- SSH: stored out-of-band for operators only
- hosted gRPC endpoint: `studio.agentslab.host:50053`

## Current server constraints

The server is good enough for repeated CPU-oriented benchmark work, but it is not
the right machine for a final `1TB` claim yet.

- about `12` CPU
- about `62 GiB` RAM
- limited free root-disk capacity for honest `1TB` benchmarking
- weak GPU (`GeForce GT 710`), so embedding work should remain CPU-friendly

Because of that, the large-scale benchmark story should stay staged:

1. `10GB`
2. `100GB`
3. `250GB`
4. `1TB` only after moving to a larger disk or attached storage

## Current benchmark workspace on the server

- `<benchmark-workspace>/datasets`
- `<benchmark-workspace>/embeddings`
- `<benchmark-workspace>/results`
- `<benchmark-workspace>/logs`
- `<benchmark-workspace>/work`

These locations should be treated as the canonical landing zone for heavy benchmark
artifacts before we selectively publish summaries back into this repo.

Local snapshots now checked into this repo:

- [`reports/runs/20260427T224143Z_scifact_baseline_pilot_metadata.json`](../reports/runs/20260427T224143Z_scifact_baseline_pilot_metadata.json)
- [`reports/runs/20260427T225122Z_scifact_baseline_summary.json`](../reports/runs/20260427T225122Z_scifact_baseline_summary.json)
- [`reports/runs/20260427T225122Z_scifact_baseline_embedding_metadata.json`](../reports/runs/20260427T225122Z_scifact_baseline_embedding_metadata.json)
- [`reports/runs/20260427T230412Z_scifact_bge_small_summary.json`](../reports/runs/20260427T230412Z_scifact_bge_small_summary.json)
- [`reports/runs/20260427T230412Z_scifact_bge_small_embedding_metadata.json`](../reports/runs/20260427T230412Z_scifact_bge_small_embedding_metadata.json)

## Established quality finding

The most important retrieval-quality result so far is that embedding choice moved
quality more than HNSW tuning on SciFact.

Latest verified server runs:

- baseline sweep run: `20260427T225122Z`
- baseline pilot metadata run: `20260427T224143Z`
- `BAAI/bge-small-en-v1.5` sweep run: `20260427T230412Z`

Baseline embedding metadata:

- backend: `sentence-transformers`
- model: `sentence-transformers/all-MiniLM-L6-v2`
- dataset: SciFact
- documents: `5183`
- queries: `300`
- dimension: `384`

BGE comparison embedding metadata:

- backend: `fastembed`
- model: `BAAI/bge-small-en-v1.5`
- dataset: SciFact
- documents: `5183`
- queries: `300`
- dimension: `384`

Summary of the current conclusion:

- baseline SciFact `recall@5` was about `0.7109`
- `BAAI/bge-small-en-v1.5` reached about `0.7624` `recall@5`
- `MRR` and `nDCG` improved as well
- latency remained in a good range for the hosted setup
- HNSW parameter sweeps did not meaningfully change quality compared with the
  embedding-model change

### Exact SciFact comparison

| Embeddings | Run | recall@5 | MRR | nDCG@5 | p50 (ms) | p95 (ms) | mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` + `fast` HNSW | `20260427T225122Z` | `0.7109` | `0.5883` | `0.6135` | `0.857` | `0.996` | `0.800` |
| `all-MiniLM-L6-v2` + `balanced` HNSW | `20260427T225122Z` | `0.7109` | `0.5883` | `0.6135` | `0.953` | `1.019` | `0.880` |
| `all-MiniLM-L6-v2` + `quality` HNSW | `20260427T225122Z` | `0.7109` | `0.5883` | `0.6135` | `0.900` | `1.001` | `0.813` |
| `BAAI/bge-small-en-v1.5` + `fast` HNSW | `20260427T230412Z` | `0.7624` | `0.6603` | `0.6812` | `0.920` | `1.041` | `0.840` |
| `BAAI/bge-small-en-v1.5` + `balanced` HNSW | `20260427T230412Z` | `0.7624` | `0.6603` | `0.6812` | `0.929` | `0.985` | `0.833` |
| `BAAI/bge-small-en-v1.5` + `quality` HNSW | `20260427T230412Z` | `0.7624` | `0.6603` | `0.6812` | `0.720` | `0.992` | `0.775` |

### Best-to-best summary

Using the best observed latency profile from each embedding set:

| Comparison | recall@5 | MRR | nDCG@5 | mean latency |
| :--- | ---: | ---: | ---: | ---: |
| baseline `all-MiniLM-L6-v2` | `0.7109` | `0.5883` | `0.6135` | `0.800 ms` |
| `BAAI/bge-small-en-v1.5` | `0.7624` | `0.6603` | `0.6812` | `0.775 ms` |

Observed gains from the embedding change:

- `recall@5`: `+0.0516` absolute, about `+7.3%` relative
- `MRR`: `+0.0719` absolute, about `+12.2%` relative
- `nDCG@5`: `+0.0677` absolute, about `+11.0%` relative

Interpretation:

- the next strong retrieval lever is embedding selection
- HNSW sweeps are still useful for latency/recall tradeoff mapping
- we should not oversell ANN tuning as the main quality breakthrough

## Recommended benchmark order from here

For retrieval work, keep the methodology disciplined:

1. fix the dataset
2. fix the embedding model
3. sweep HNSW settings
4. compare `recall@k`, `MRR`, `nDCG`, and latency
5. compare embedding models on the same benchmark path

For large-scale system work:

1. complete clean `10GB` results
2. publish `100GB` results
3. run `250GB` only after confirming disk headroom
4. defer `1TB` until storage is expanded

## Scripts that define the server lane

- `scripts/run_sochdb_grpc_quality_sweep.sh`
- `scripts/run_sochdb_embedding_bakeoff.sh`

Related planning docs:

- [`STAGED_BENCHMARK_PLAN.md`](./STAGED_BENCHMARK_PLAN.md)
- [`RETRIEVAL_AND_VECTOR_PLAN.md`](./RETRIEVAL_AND_VECTOR_PLAN.md)

## What is still pending

- add the next embedding-model comparison beyond `BAAI/bge-small-en-v1.5`
- complete the staged `10GB` -> `100GB` -> `250GB` scale path
- defer `1TB` claims until storage is expanded

This file should be the first place to update whenever new server benchmark work
changes the current benchmark story.
