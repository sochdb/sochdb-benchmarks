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

> **Note**: Results measured using `benchmarks/comprehensive_benchmark.py`. See "Run Environment" below for hardware details.

## Why Search Latency Matters More Than Insert

In **Agentic AI** and **RAG** workflows, the performance bottleneck typically lies in the "read" path (retrieval), not the "write" path (ingestion).

1.  **Blocking the Thought Loop**: When an AI Agent "thinks", it queries its memory. This is a blocking operation. Every millisecond of latency delays the agent's next action. ToonDB's **0.3ms** latency keeps the thought loop tight and responsive.
2.  **Frequency of Operations**: An agent might query its memory dozens of times per turn (e.g., planning, reflecting, tool selection). Ingestion usually happens once (e.g., loading a document) or asynchronously in the background.
3.  **Real-Time Interaction**: For user-facing chatbots, retrieval latency adds directly to the "Time to First Token". Columnar stores like LanceDB prioritize ingestion speed (good for offline training), but ToonDB prioritizes retrieval speed (critical for online interaction).

**Verdict**: If your app waits for the database to answer before it can reply to the user, optimize for **Search Latency** (ToonDB). If you are batch-processing terabytes of logs overnight, optimize for **Insert Rate** (LanceDB).

## Detailed Comparison

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

======================================================================
   PERFORMANCE ANALYSIS
======================================================================

  Highest Insert Rate: LANCEDB (96852 vec/s)
  Lowest Search Latency: TOONDB (0.325ms)

  Relative Performance (ToonDB baseline):
    vs chromadb  : Insert 0.23x    | Search 2.11x    (relative speedup)
    vs duckdb    : Insert 0.61x    | Search 2.78x    (relative speedup)
    vs lancedb   : Insert 0.02x    | Search 12.54x   (relative speedup)
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

3. **Real-World Embedding Test**:
   Requires Azure OpenAI keys in `.env`. Benchmarks end-to-end latency.
   ```bash
   python3 benchmarks/real_embedding_benchmark.py
   ```
