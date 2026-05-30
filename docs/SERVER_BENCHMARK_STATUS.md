# Server Benchmark Status

This document captures the current benchmark story for the hosted SochDB server.

## Current setup

- heavy benchmark work runs on the benchmark server, not on laptops
- hosted gRPC demo endpoint: `studio.agentslab.host:50053`
- current server class: about `12` CPU, about `62 GiB` RAM
- current storage is not appropriate for a final `1TB` claim yet

Because of that, large-scale benchmarking stays staged:

1. `10GB`
2. `100GB`
3. `250GB`
4. `1TB` only after storage expansion

## Published quality result

Quality is measured separately from scale, using SciFact retrieval benchmarks.

Current best published result:

- model: `BAAI/bge-base-en-v1.5`
- recall@5: `0.8121`
- MRR: `0.7017`
- nDCG@5: `0.7258`

Main takeaway:

- embedding choice moved quality more than HNSW tuning in the current setup

Useful reference points:

| Embeddings | recall@5 | MRR | nDCG@5 | mean latency |
| :--- | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | `0.7109` | `0.5883` | `0.6135` | `0.800 ms` |
| `BAAI/bge-small-en-v1.5` | `0.7624` | `0.6603` | `0.6812` | `0.775 ms` |
| `thenlper/gte-small` | `0.7786` | `0.6711` | `0.6944` | `0.878 ms` |
| `BAAI/bge-base-en-v1.5` | `0.8121` | `0.7017` | `0.7258` | `1.951 ms` |

Published artifacts:

- [`20260427T225122Z_scifact_baseline_summary.json`](../reports/runs/20260427T225122Z_scifact_baseline_summary.json)
- [`20260427T230412Z_scifact_bge_small_summary.json`](../reports/runs/20260427T230412Z_scifact_bge_small_summary.json)
- [`20260429T_next_gte_small_st_summary.json`](../reports/runs/20260429T_next_gte_small_st_summary.json)
- [`20260429T_next_bge_base_summary.json`](../reports/runs/20260429T_next_bge_base_summary.json)

## Published `10GB` scale result

Dataset:

- run family: `synthetic_10gib_768d`
- vectors: `3,495,253`
- dimension: `768`

Build result:

- build throughput: about `891.6 vec/s`
- build time: about `3920.14 s`
- output index size: about `10,069.1 MB`

Corrected steady-state search result:

- one-time index load: about `106.85 s`
- sequential search: about `506.63 QPS`
- sequential mean latency: about `1.97 ms`
- sequential `p50`: about `1.87 ms`
- sequential `p95`: about `2.40 ms`
- batch search: about `356 QPS`

Important note:

- the earlier `0.0091 QPS` / `~110s per query` result came from a bad benchmark
  harness path and should not be treated as the real steady-state engine result

Published artifacts:

- [`20260503T_stage10gb_d768_sochdb_vector_summary.json`](../reports/runs/20260503T_stage10gb_d768_sochdb_vector_summary.json)
- [`20260503T_stage10gb_d768_metadata.json`](../reports/runs/20260503T_stage10gb_d768_metadata.json)
- [`20260512T_10gb_optimized_native_summary.json`](../reports/runs/20260512T_10gb_optimized_native_summary.json)
- [`20260512T_10gb_optimized_native_metadata.json`](../reports/runs/20260512T_10gb_optimized_native_metadata.json)

## What is pending

- publish `100GB` results using the corrected native steady-state methodology
- publish `250GB` results after confirming disk headroom
- defer `1TB` claims until storage is expanded
