# SochDB 360° Performance Report

Generated: 2026-01-13 00:14:33


## 1. Retrieval Quality

| Metric | Value |
|--------|-------|
| Recall@1 | 0.630 |
| Recall@5 | 0.622 |
| Recall@10 | 0.636 |
| Recall@20 | 0.583 |
| Recall@100 | 0.678 |
| MRR | 1.000 |
| NDCG@10 | 0.748 |

## 2. Query Latency

| Condition | p50 (ms) | p95 (ms) | p99 (ms) |
|-----------|----------|----------|----------|
| cold | 0.280 | 0.410 | 0.514 |
| warm | 0.255 | 0.350 | 0.447 |
| k=1 | 0.308 | 0.461 | 0.583 |
| k=10 | 0.302 | 0.374 | 0.399 |
| k=100 | 0.493 | 0.535 | 0.575 |

## 3. Throughput

- Single-thread QPS: 3,216
- 4-thread QPS: 5,197

## 4. Ingestion

- Insert rate: 2,477 vec/s
- Time-to-searchable: 0.65ms

## 5. Resource Efficiency

- RAM per vector: 93.4 bytes
- Index size: 0.9 MB

## 6. Agent Memory Performance

- Read recall: 98.00%
- Staleness error rate: 100.00%
- Memory latency overhead: 0.143ms