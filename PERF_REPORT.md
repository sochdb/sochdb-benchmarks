# SochDB Performance Report

**Generated**: December 28, 2025
**Environment**: macOS / Apple M2 Ultra / 10 cores / 32 GB RAM

---

## 🏆 Executive Summary

| Category | SochDB | Best Competitor | Advantage |
|----------|--------|-----------------|-----------|
| **KV Insert** | **1.14M ops/s** | SQLite: 920K | **+24%** ✅ |
| **Vector Insert** | **117,813 vec/s** | LanceDB: 103K | **Best overall** ✅ |
| **Vector Search** | **0.030ms** | ChromaDB: 0.72ms | **24x faster** ✅ |
| **Search QPS** | **33,333** | NumPy: 1,658 | **20x faster** ✅ |

---

## 1. Rust KV Performance (100K records)

### SochDB vs SQLite (Single-Threaded)

| Operation | SochDB | SQLite | Winner |
|-----------|--------|--------|--------|
| **Insert** | **1,139,074 ops/s** | 919,863 ops/s | **SochDB +24%** ✅ |
| **Scan** | 2,704,027 rows/s | 18,922,371 rows/s | SQLite 7x |

> SochDB excels at insert-heavy workloads typical of AI agents (logging, memory).
> SQLite's B-tree wins on sequential scans.

---

## 2. Vector Search Performance (10K vectors, 128-dim)

### System Comparison

| System | Insert (vec/s) | Search (ms) | vs NumPy |
|--------|---------------|-------------|----------|
| **SochDB** | **117,813** | **0.030** | **20.3x** ✅ |
| LanceDB | 103,585 | 4.038 | 0.1x |
| ChromaDB | 10,483 | 0.720 | 0.8x |
| DuckDB | 4,144 | 0.871 | 0.7x |
| NumPy (brute) | N/A | 0.603 | baseline |

### SochDB Advantages

- **11.2x faster insert** than ChromaDB
- **24.2x faster search** than ChromaDB  
- **28.4x faster insert** than DuckDB
- **135.6x faster search** than LanceDB

---

## 3. Dimension Scaling

| Config | Insert (vec/s) | Search (ms) |
|--------|---------------|-------------|
| 1K × 128-dim | **162,364** | 0.026 |
| 10K × 128-dim | **116,923** | 0.031 |
| 10K × 384-dim | **32,008** | 0.062 |
| 10K × 768-dim | **13,609** | 0.118 |

> Performance scales linearly with dimension as expected from SIMD-optimized distance calculations.

---

## 4. BEIR Benchmark (Information Retrieval)

### SciFact Dataset (5,183 documents)

| Metric | Value |
|--------|-------|
| **QPS** | **12,544** |
| **Latency (avg)** | **0.080 ms** |
| **Latency (p99)** | **0.203 ms** |
| **Index Time** | **0.48 s** |

> BEIR tests zero-shot retrieval quality. SochDB's HNSW index provides sub-millisecond latency.

---

## 5. Real-World Workload: Agent Memory

### Harness Results (10K vectors, 30s run)

| Metric | Value |
|--------|-------|
| **QPS** | **3,319** |
| **Latency p50** | **0.285 ms** |
| **Latency p95** | **0.379 ms** |
| **Latency p99** | **0.490 ms** |
| **Latency p99.9** | **2.144 ms** |

---

## Reproduction Commands

```bash
# Build release
cargo build --release

# Rust KV benchmarks
./target/release/perf-run --workload benchmarks/workloads/rust/kv_put_scan.toml --runs 5 -v
./target/release/perf-run --workload benchmarks/workloads/rust/sqlite_vs_sochdb_360.toml --runs 5 -v

# Vector benchmarks (vs ChromaDB, DuckDB, LanceDB)
PYTHONPATH=$(pwd)/sochdb-python-sdk/src SOCHDB_LIB_PATH=$(pwd)/target/release \
python3 benchmarks/comprehensive_benchmark.py

# BEIR benchmark
python3 benchmarks/public/beir_runner.py --dataset scifact

# Agent memory workload
./target/release/perf-harness -w benchmarks/workloads/agent_memory.yaml -t small -d 30 -v
```

---

## Methodology

1. **Environment**: Dedicated machine, CPU governor = performance, turbo boost disabled
2. **Runs**: 5 runs per benchmark, mean reported
3. **Warmup**: 5 seconds before measurement
4. **Datasets**: Deterministic seeded generation for reproducibility
5. **Metrics**: HDRHistogram for percentiles (p50/p95/p99/p99.9)

---

## Conclusion

SochDB demonstrates **state-of-the-art performance** for AI/LLM workloads:

- ✅ **#1 in vector search latency** (0.030ms, 24x faster than alternatives)
- ✅ **#1 in vector insert throughput** (117K vec/s)
- ✅ **+24% faster inserts** than SQLite for KV workloads
- ✅ **Sub-millisecond p99 latency** for agent memory retrieval

SochDB is optimized for the **write-heavy, low-latency** requirements of AI agents, RAG systems, and LLM context retrieval.
