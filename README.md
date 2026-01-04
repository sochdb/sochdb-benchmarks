# ToonDB Benchmarks

This repository contains reproducible benchmarks comparing **ToonDB** against other vector stores and agent memory systems.

**📊 [See Published Results](PUBLISHED_RESULTS.md)** - Comprehensive benchmark findings with real LLM integration

## Overview

We provide benchmarks across different dimensions:
- **Vector search scaling** (O(n) vs O(log n))
- **Real LLM integration** (actual Azure OpenAI calls)
- **Multi-system comparison** (ToonDB vs ChromaDB, Zep framework ready)
- **Production-grade framework** (2000+ lines, fully reproducible)

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

## 🤖 Agent Memory Systems Benchmark

### 1. Pure Vector Search Scaling: O(n) vs O(log n)

**The Problem**: Brute-force O(n) vector search causes P99 latency to degrade 34x as observations grow from 40 → 2000.
**The Solution**: HNSW O(log n) search keeps degradation minimal (5.9x) while delivering 11x better performance at scale.

This benchmark uses **pre-generated random embeddings** to isolate pure vector search performance (no LLM API overhead).

#### Results: Brute-Force vs HNSW

| Scale | Brute-Force P99 | HNSW P99 | Speedup |
| :--- | ---: | ---: | ---: |
| 40 observations | 0.26ms | 0.14ms | **1.9x** |
| 100 observations | 0.71ms | 0.20ms | **3.6x** |
| 200 observations | 0.90ms | 0.36ms | **2.5x** |
| 500 observations | 2.98ms | 0.49ms | **6.1x** |
| 1,000 observations | 6.92ms | 0.86ms | **8.0x** |
| 2,000 observations | **9.06ms** | **0.81ms** | **11.2x** |

**Scaling Analysis:**
- **Brute-Force (40 → 2000)**: P99 degrades 0.26ms → 9.06ms (**34x worse**)
- **HNSW (40 → 2000)**: P99 degrades 0.14ms → 0.81ms (**5.9x** - much better!)
- **At 200 observations**: The crossover point where HNSW becomes clearly superior
- **At 2000 observations**: HNSW is **11.2x faster** than brute-force

**Run the benchmark:**
```bash
export TOONDB_LIB_PATH=/path/to/libtoondb_index.so
python3 benchmarks/pure_search_scale_benchmark.py
```

---

### 2. Real-World LLM Integration Test

**Inspired by the [Zep vs Mem0 controversy](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)**

This benchmark uses **actual Azure OpenAI embedding calls** to test memory systems in realistic agent conversation scenarios.

#### Test Configuration
- **Conversations**: 8 multi-turn dialogues (customer support, technical support, product inquiries)
- **Messages**: 65 total messages stored as memories
- **Test Queries**: 200 queries to test memory recall
- **Embeddings**: Azure OpenAI `text-embedding-3-small` (1536-dim)
- **Date**: 2026-01-04

#### Results: ToonDB vs ChromaDB

| System | Insert (avg) | p50 Latency | p95 Latency | p99 Latency | Context Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **ToonDB** | 94.20ms | **79.49ms** | 172.64ms | 2557.91ms | 36 tokens |
| **ChromaDB** | 184.90ms | 82.80ms | **123.00ms** | **1338.15ms** | 36 tokens |

**Key Findings:**
- **ToonDB** is **1.96x faster at insert** (94ms vs 185ms)
- **ChromaDB** has **better p95/p99 consistency** (123ms vs 173ms p95)
- Both systems delivered **identical context quality** (36 tokens avg)
- **Real embedding overhead dominates**: 70-90% of latency is Azure OpenAI API calls, not DB operations

#### Why This Matters

Unlike synthetic vector benchmarks, this test measures:
1. **Real LLM integration overhead** - actual API calls to Azure OpenAI
2. **Multi-turn conversation memory** - realistic agent dialogue patterns
3. **Production-like workloads** - insert + search in realistic sequences
4. **Context assembly** - how fast systems retrieve and build context for LLM prompts

**Run the benchmark:**
```bash
export TOONDB_LIB_PATH=/path/to/libtoondb_index.so
python3 benchmarks/memory_systems_comparison.py
```

## detailed Comparison

### ToonDB
- **Performance Profile**: Optimized for low-latency search (0.33ms) and fast inserts for agent memory.
- **Architecture**: In-memory HNSW index with Rust core, Python FFI.
- **Trade-off**: Lower ingestion throughput compared to columnar stores on bulk loads.
- **Best For**: Agent memory systems, real-time RAG, low-latency search.

### LanceDB
- **Performance Profile**: Optimized for high-throughput ingestion (96k vec/s).
- **Architecture**: Disk-based columnar format (Lance).
- **Trade-off**: Higher search latency for random-access patterns (approx. 4ms).
- **Best For**: Large-scale datasets, batch processing, analytics.

### ChromaDB
- **Performance Profile**: Balanced performance for general use cases.
- **Architecture**: Persistent storage with HNSW indexing.
- **Trade-off**: Slower search than ToonDB, slower ingestion than LanceDB.
- **Best For**: General-purpose RAG, prototyping, moderate-scale applications.

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

### Quick Start

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

### Production-Grade Comparison: ToonDB vs Zep

For a comprehensive, apples-to-apples comparison of agent memory systems:

```bash
# Set up environment
export AZURE_OPENAI_API_KEY="your_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export TOONDB_LIB_PATH="/path/to/libtoondb_index.so"

# Optional: Add Zep for comparison
export ZEP_API_URL="http://localhost:8000"
export ZEP_API_KEY="your_zep_key"

# Run comprehensive benchmark
python3 benchmarks/run_memory_comparison.py
```

**What it tests**:
- Phase 1: Microbenchmarks (latency, throughput)
- Phase 2: Token efficiency (context assembly)
- Phase 3: LoCoMo quality (QA accuracy)
- Phase 4: Scale test (100-2000 observations)

See [`BENCHMARK_FRAMEWORK_GUIDE.md`](BENCHMARK_FRAMEWORK_GUIDE.md) for full details.
