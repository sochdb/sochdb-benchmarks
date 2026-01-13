# SochDB vs Zep Memory System Comparison

## Overview

This is a comprehensive, **apples-to-apples** benchmark comparing SochDB and Zep for agent memory systems.

Both systems are evaluated on the same job:
> Given user/session state + history + business docs, return the best context block for the next LLM call under a token budget.

## Benchmark Design

### Fair Comparison Principles

1. **Same Interface**: Both systems implement the same `MemorySystemAdapter` interface
2. **Same Embeddings**: Both use identical embedding model (text-embedding-3-small, 1536-dim)
3. **Same LLM**: Both evaluated with same model, prompt, temperature
4. **Same Datasets**: Both tested on identical workloads
5. **Transparent Setup**: All configurations documented and reproducible

### Test Matrix

| System | Config | Mode |
|--------|--------|------|
| SochDB | Default (m=16, ef=100) | Local |
| SochDB | Tuned (optimized params) | Local |
| Zep | Default/Recommended | Local or Cloud |
| Zep | Tuned | Local or Cloud |

## Architecture

```
memory_benchmark_harness.py       # Core interfaces and data models
├── MemorySystemAdapter           # Abstract base class
├── Message, Document             # Data models
├── ContextResult                 # Retrieval result
└── BenchmarkMetrics              # Metrics collection

adapters/
├── sochdb_adapter.py             # SochDB implementation
│   └── Uses: HNSW index + hierarchical storage
└── zep_adapter.py                # Zep implementation
    └── Uses: Zep Memory API + Context Block

workload_generator.py
├── SyntheticWorkloadGenerator    # Production-like workload
├── LoCoMoDatasetLoader           # Quality benchmark
└── QueryGenerator                # Test queries

run_memory_comparison.py          # Main benchmark orchestrator
├── Phase 1: Microbenchmarks
├── Phase 2: Token Efficiency
├── Phase 3: LoCoMo Quality
└── Phase 4: Scale Test
```

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install SochDB (if not already installed)
pip install sochdb

# 3. Optional: Install Zep for comparison
pip install zep-python

# 4. Set up Zep (optional)
# For local Zep:
docker run -d -p 8000:8000 ghcr.io/getzep/zep:latest

# For Zep Cloud:
# Sign up at https://www.getzep.com/
```

## Configuration

Set environment variables:

```bash
# Required: Azure OpenAI (for embeddings)
export AZURE_OPENAI_API_KEY="your_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"

# Required: SochDB library path
export SOCHDB_LIB_PATH="/path/to/libsochdb_index.so"

# Optional: Zep (for comparison)
export ZEP_API_URL="http://localhost:8000"  # or Zep Cloud URL
export ZEP_API_KEY="your_zep_key"  # for Zep Cloud
```

## Running Benchmarks

### Quick Start

```bash
# Run full benchmark suite
python3 benchmarks/run_memory_comparison.py
```

### Phase-by-Phase

```bash
# Run specific phases (edit script to enable/disable)
python3 benchmarks/run_memory_comparison.py
```

## Benchmark Phases

### Phase 1: Microbenchmarks

**What it tests**: Pure system performance

- **Ingest throughput**: messages/sec, docs/sec
- **Retrieval latency**: p50/p95/p99 for:
  - Recent memory query
  - Old memory query
  - Multi-hop query

**Metrics**:
- Latency (ms)
- Throughput (ops/sec)

### Phase 2: Token Efficiency

**What it tests**: Context assembly within token budgets

For each token budget (2k / 4k / 8k):
- Retrieve context
- Measure actual tokens used
- Track truncation events

**Metrics**:
- Avg/p95 tokens per call
- Truncation rate
- Budget utilization

### Phase 3: LoCoMo Quality

**What it tests**: End-to-end QA accuracy

- Ingest LoCoMo conversations
- Run QA tasks
- Measure answer quality

**Metrics**:
- QA accuracy (% correct)
- Summarization quality
- Hallucination rate

### Phase 4: Scale Test

**What it tests**: Performance under load

Test scales: 100, 500, 1000, 2000 observations

**Metrics**:
- p50/p95/p99 latency at each scale
- Degradation factor
- Memory usage

## Metrics Collected

### A) Context Payload Efficiency

- **Prompt tokens** (mean / p95) for final LLM call
- **Context tokens** contributed by memory system
- **Truncation rate**: how often context exceeds budget

### B) Retrieval Performance

- **retrieve_context() latency** p50/p95/p99
- **QPS** at fixed concurrency
- **Tail latency** under load

### C) End-to-End Quality

- **LoCoMo QA accuracy**
- **Hallucination / contradiction rate**

### D) Operational Correctness

- **Durability** on crash/restart
- **Update/delete correctness**
- **Multi-tenant isolation**

## Results Format

Results are saved to `benchmark_results/comparison_results_TIMESTAMP.json`:

```json
{
  "phase1_microbenchmarks": {
    "SochDB": {
      "ingest_avg_ms": 94.2,
      "retrieval_p50_ms": 79.5,
      "retrieval_p95_ms": 172.6,
      "retrieval_p99_ms": 2557.9,
      "avg_tokens": 36
    },
    "Zep": { ... }
  },
  "phase2_token_efficiency": { ... },
  "phase3_locomo_quality": { ... },
  "phase4_scale_test": { ... }
}
```

## Interpreting Results

### Latency

- **p50**: Typical case performance
- **p95**: Most user experiences
- **p99**: Tail latency (affects agent UX)

**Target**: p95 < 200ms for good UX (Zep claims sub-200ms)

### Token Efficiency

- **Lower is better** for API costs
- **Budget utilization**: how well system uses available tokens
- **Truncation rate**: reliability of context assembly

### Quality

- **QA accuracy**: does the system retrieve relevant context?
- **Consistency**: does performance degrade over time?

### Scale

- **Degradation factor**: how much slower at 2000 vs 100 observations
- **SochDB target**: O(log n) HNSW should keep degradation minimal
- **Zep target**: Should maintain sub-200ms p95 at scale

## Fairness Guardrails

To ensure nobody can dispute results:

1. ✅ Same embedding model + dimension (text-embedding-3-small, 1536)
2. ✅ Same top-k and score thresholds (or clearly justified)
3. ✅ Same LLM + prompt + temperature for evaluation
4. ✅ Disclosed if Zep is Cloud and SochDB is local
5. ✅ Both systems use default/recommended settings
6. ✅ All source code + configs + datasets included

## Reproducibility

Everything needed to reproduce:

```
benchmarks/
├── memory_benchmark_harness.py     # Core framework
├── adapters/
│   ├── sochdb_adapter.py           # SochDB implementation
│   └── zep_adapter.py              # Zep implementation
├── workload_generator.py           # Dataset generators
├── run_memory_comparison.py        # Main runner
└── MEMORY_COMPARISON_README.md     # This file

benchmark_results/
└── comparison_results_*.json       # Raw results
```

## Expected Performance

Based on design and previous benchmarks:

### SochDB

- **Retrieval p95**: 50-200ms (HNSW O(log n))
- **Scale degradation**: Minimal (5-10x at 2000 obs)
- **Token efficiency**: Direct TOON output
- **Best for**: Low-latency, agent memory, real-time RAG

### Zep

- **Retrieval p95**: <200ms (claimed)
- **Scale degradation**: Unknown (to be measured)
- **Token efficiency**: Automated context assembly
- **Best for**: Full-featured agent memory, graph-based memory

## Troubleshooting

### SochDB Issues

```bash
# Library not found
export SOCHDB_LIB_PATH=/usr/local/lib/python3.11/site-packages/sochdb/lib/x86_64-unknown-linux-gnu/libsochdb_index.so

# Check installation
python3 -c "import sochdb; print(sochdb.__version__)"
```

### Zep Issues

```bash
# Local server not responding
docker ps  # Check if Zep is running
curl http://localhost:8000/healthz

# Cloud connection
echo $ZEP_API_URL  # Should be set
echo $ZEP_API_KEY  # Should be set
```

### Azure OpenAI Issues

```bash
# Test connection
python3 -c "from openai import AzureOpenAI; client = AzureOpenAI(api_key='$AZURE_OPENAI_API_KEY', azure_endpoint='$AZURE_OPENAI_ENDPOINT', api_version='2024-12-01-preview'); print(client.models.list())"
```

## References

- [SochDB Documentation](https://docs.rs/sochdb)
- [Zep Documentation](https://docs.getzep.com)
- [LoCoMo Benchmark](https://github.com/snap-research/LoCoMo)
- [Zep vs Mem0 Blog Post](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)

## Contributing

To add a new memory system:

1. Implement `MemorySystemAdapter` interface
2. Add adapter to `adapters/`
3. Register in `run_memory_comparison.py`
4. Run benchmarks
5. Submit PR with results

## License

MIT License - See LICENSE file for details
