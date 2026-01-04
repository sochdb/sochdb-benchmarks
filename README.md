# ToonDB Benchmarks

This repository contains reproducible benchmarks comparing **ToonDB** against other vector stores: **ChromaDB**, **LanceDB**, **DuckDB**, and **SQLite (VSS)**.

The goal is to provide a comprehensive view of performance across different workloads: write-heavy, read-heavy, memory-constrained, and on-disk persistence.

## Performance Snapshot

**Scenario**: 10,000 vectors, 128-dimensions, running on local hardware.

| Database | Insert Rate | Search Latency (Avg) | Storage Engine | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **ToonDB** | ~2,377 vec/s | **0.325 ms** | In-Memory (Rust) + WAL | Low-Latency Search, Agent Memory |
| **LanceDB** | **96,852 vec/s** | 4.07 ms | Disk-Based (Lance) | Large Datasets, High-Throughput Ingestion |
| **ChromaDB** | ~10,500 vec/s | 0.69 ms | In-Memory / SQLite | General Purpose RAG, Prototyping |
| **DuckDB** | ~3,900 vec/s | 0.90 ms | OLAP + VSS | Analytical + Vector Search Hybrid |
| **NumPy** | N/A | 0.62 ms | In-Memory (Exact) | Baseline comparison |

## Benchmark Run (OpenAI Codex Environment)

**Run Date (UTC)**: 2026-01-04 03:51 UTC  
**Host**: Linux 6.12.13 (x86_64, KVM)  
**CPU**: Intel(R) Xeon(R) Platinum 8370C @ 2.80GHz (3 vCPU)  
**Command**:
```bash
TOONDB_LIB_PATH=/root/.pyenv/versions/3.12.12/lib/python3.12/site-packages/toondb/lib/x86_64-unknown-linux-gnu/libtoondb_index.so \
  python3 benchmarks/comprehensive_benchmark.py
```

### Standard Benchmark (10K vectors, 128-dim)

| System | Insert (vec/s) | Search (ms avg) | Notes |
| :--- | ---: | ---: | :--- |
| NumPy (brute-force) | N/A | 0.359 | Baseline |
| ChromaDB | 3,925 | 1.548 | — |
| LanceDB | 18,245 | 14.039 | IVF-PQ index |
| ToonDB | 2,442 | 0.580 | Rust HNSW via Python FFI |

### Multi-Configuration (ToonDB)

| Config | Insert (vec/s) | Search (ms avg) |
| :--- | ---: | ---: |
| 1,000 × 128 | 6,455 | 0.250 |
| 10,000 × 128 | 2,240 | 0.567 |
| 10,000 × 384 | 371 | 3.634 |
| 10,000 × 768 | 1,334 | 1.197 |

### Errors / Limitations

* **SQLite-VSS**: `sqlite3.Connection` in this environment does not expose `enable_load_extension`, so the benchmark could not load the extension.
* **DuckDB VSS**: Extension download failed (no access to `extensions.duckdb.org` from the container).

## 🏗️ Systems Engineering Evaluation

Beyond microbenchmarks, we stress-tested ToonDB's "Actual" production capability for Agentic workloads. 

### 1. The "Agent Loop" Macrobenchmark
We simulated a long-running agent conversation where the system must simultaneously **Write** new observations and **Read/Assemble** context for a prompt.

| Metric (P99 Latency) | ToonDB (Unified) | SQLite + Chroma (Fragmented) | Improvement |
| :--- | :--- | :--- | :--- |
| **Write (Append)** | **0.01 ms** | 2.80 ms | **280x** Faster |
| **Read (Context)** | **0.01 ms** | 3.06 ms | **300x** Faster |

> **Why This Matters**: ToonDB acts as an integrated memory layer. The "Fragmented" baseline requires network/IPC hops between Python, SQLite, and Chroma. ToonDB keeps the "Thought Loop" tight.

### 2. Transactional Integrity (Crash Test)
We subjected ToonDB to a "Jepsen-lite" test: heavily writing to a key and randomly force-killing the process (`kill -9`).

- **Result**: ✅ PASSED
- **Recovery Time**: 4.31 ms
- **Consistency**: No data corruption; WAL successfully replayed last committed transaction.

### 3. Hardware Efficiency (Microbenchmark)
We isolated the cosine distance kernel to check SIMD usage on ARM (Apple M1 Max).

- **Finding**: Raw kernel throughput via FFI is lower than NumPy (0.08x) due to Python<->Rust boundary overhead on single queries.
- **Verdict**: ToonDB is optimal for **Search** (where work stays in Rust) but has high overhead for basic vector math ops in Python compared to highly optimized BLAS.

## detailed Comparison

### ToonDB
- **Performance Profile**: Optimized for low-latency search (0.33ms).
- **Architecture**: In-memory HNSW index with Rust core.
- **Trade-off**: Lower ingestion throughput compared to columnar stores.

### LanceDB
- **Performance Profile**: Optimized for high-throughput ingestion (96k vec/s).
- **Architecture**: Disk-based columnar format (Lance).
- **Trade-off**: Higher search latency for random-access patterns (approx. 4ms).

### ChromaDB
- **Performance Profile**: Balanced performance for general use cases.
- **Architecture**: Persistent storage with HNSW indexing.
- **Trade-off**: Slower search than ToonDB, slower ingestion than LanceDB.

## Verification

**Run Environment**:
- **Hardware**: Mac Studio (Apple M1 Max, 32GB RAM)
- **OS**: macOS 26.2
- **Date**: January 03, 2026
- **Command**: `python3 benchmarks/comprehensive_benchmark.py`

### Raw Output Log (Excerpt)

Below is the output from the strictly verified benchmark run:

```text
======================================================================
   FINAL SUMMARY
======================================================================

System                    Insert (vec/s)     Search (ms)     Speedup vs NumPy
---------------------------------------------------------------------------
NumPy (brute-force)       N/A                0.619           1.0x (baseline)
ChromaDB                  10558              0.687           0.9x
DuckDB                    3886               0.904           0.7x
LanceDB                   96852              4.074           0.2x
ToonDB                    2377               0.325           1.9x
```

## Running the Benchmarks

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Comprehensive Suite**:
   Runs all DBs against synthetic data (10k-100k vectors).
   ```bash
   python3 benchmarks/comprehensive_benchmark.py
   ```

3. **Run Systems Evaluation**:
   ```bash
   python3 benchmarks/macro_agent_benchmark.py
   python3 benchmarks/crash_test.py
   ```
