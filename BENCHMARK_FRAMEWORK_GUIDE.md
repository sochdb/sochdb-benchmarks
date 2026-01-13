# SochDB vs Zep: Production-Grade Benchmark Framework

## Overview

This is a comprehensive, **apples-to-apples** benchmark framework for comparing agent memory systems (SochDB vs Zep).

Unlike previous benchmarks that used synthetic vectors, this framework:
- ✅ Uses the **same interface** for both systems
- ✅ Tests on **real-world workloads** (LoCoMo + synthetic production data)
- ✅ Measures **what actually matters** (latency, tokens, quality)
- ✅ Is **fully reproducible** (all code + configs included)

## The Problem We're Solving

The [Zep vs Mem0 controversy](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) highlighted issues with memory system benchmarks:

1. **Flawed implementations**: Mem0's benchmark used incorrect Zep configurations
2. **Synthetic data**: Didn't reflect real agent workloads
3. **Cherry-picked metrics**: Focused on favorable comparisons
4. **Not reproducible**: Hard to verify claims

## Our Solution: Rigorous Benchmark Design

### 1. Unified Interface (Apples-to-Apples)

Both systems implement the same `MemorySystemAdapter` contract:

```python
class MemorySystemAdapter(ABC):
    def ingest_messages(user_id, session_id, messages[]) -> latency
    def ingest_docs(tenant_id, docs[]) -> latency
    def retrieve_context(user_id, session_id, query, token_budget) -> ContextResult
    def update_memory(...) -> success
    def delete_memory(...) -> success
```

**What this ensures**:
- Same inputs, same outputs
- Fair comparison of core functionality
- No implementation bias

### 2. Real-World Datasets

**A) LoCoMo-Style Conversational Memory**
- Long-term conversations (50+ turns)
- QA + event summarization tasks
- Tests memory recall quality

**B) Synthetic Production Workload**
- 10k users × 10 sessions × 200 messages
- 100k knowledge documents
- Mix of short/long facts + preference changes

**What this tests**:
- Scale performance (not just toy examples)
- Multi-tenant isolation
- Temporal reasoning (facts change over time)

### 3. Comprehensive Metrics

**Context Payload Efficiency**:
- Total tokens per LLM call (mean / p95)
- Context tokens from memory system
- Truncation rate (budget exceeded)

**Retrieval Performance**:
- Latency p50/p95/p99
- QPS at fixed concurrency
- Tail latency under load

**End-to-End Quality**:
- LoCoMo QA accuracy
- Hallucination rate
- Temporal consistency

**Operational Correctness**:
- Durability (crash recovery)
- Update/delete correctness
- Multi-tenant isolation

### 4. Four Benchmark Phases

**Phase 1: Microbenchmarks** (Pure Performance)
- Ingest throughput
- Retrieval latency (recent, old, multi-hop queries)
- Baseline performance measurement

**Phase 2: Token Efficiency** (What Actually Costs Money)
- Context assembly within token budgets (2k/4k/8k)
- Token usage distribution
- Truncation behavior

**Phase 3: LoCoMo Quality** (Does It Actually Work?)
- End-to-end QA tasks
- Answer quality vs. latency tradeoff
- Long-term memory consistency

**Phase 4: Scale Test** (Production Realism)
- Performance at 100, 500, 1000, 2000 observations
- Degradation factor (O(n) vs O(log n))
- Memory footprint

## Architecture

```
benchmarks/
├── memory_benchmark_harness.py          # Core framework
│   ├── MemorySystemAdapter              # Abstract interface
│   ├── Message, Document                # Data models
│   ├── ContextResult                    # Retrieval result
│   └── BenchmarkMetrics                 # Metrics collection
│
├── adapters/
│   ├── sochdb_adapter.py                # SochDB implementation
│   │   ├── HNSW index (O(log n))
│   │   ├── Hierarchical storage
│   │   └── Token-aware assembly
│   │
│   └── zep_adapter.py                   # Zep implementation
│       ├── Zep Memory API
│       ├── Context Block
│       └── Graph search
│
├── workload_generator.py
│   ├── SyntheticWorkloadGenerator       # Production-like data
│   ├── LoCoMoDatasetLoader              # Quality benchmark
│   └── QueryGenerator                   # Test queries
│
├── run_memory_comparison.py             # Main orchestrator
│   ├── Phase 1: Microbenchmarks
│   ├── Phase 2: Token Efficiency
│   ├── Phase 3: LoCoMo Quality
│   └── Phase 4: Scale Test
│
└── MEMORY_COMPARISON_README.md          # Full documentation
```

## Expected Performance Characteristics

### SochDB

**Strengths**:
- **Low latency**: HNSW O(log n) search should deliver p95 < 100ms at scale
- **Consistent scaling**: Minimal degradation (5-10x) from 100 → 2000 observations
- **Direct control**: Explicit token budgeting and context assembly

**Tradeoffs**:
- Lower ingestion throughput than batch-optimized systems
- In-memory storage (not ideal for massive archives)

**Best For**: Real-time agent memory, low-latency RAG, production agents

### Zep

**Strengths**:
- **Automated context assembly**: Graph-based memory with smart summaries
- **Rich features**: Facts, entities, summaries, temporal reasoning
- **Full-featured**: Sessions, collections, metadata management

**Tradeoffs**:
- Network overhead if using cloud
- Token efficiency depends on auto-assembly quality
- Claims sub-200ms p95 (to be verified)

**Best For**: Full-featured agent memory, complex temporal reasoning, production agents

## Fairness Guardrails

To ensure nobody can dispute results:

| Guardrail | Implementation |
|-----------|----------------|
| **Same embeddings** | Both use `text-embedding-3-small` (1536-dim) |
| **Same LLM** | Both evaluated with same model + prompt + temperature |
| **Same datasets** | Identical conversations and documents for both |
| **Same metrics** | Unified BenchmarkMetrics collection |
| **Same token budgets** | 2k/4k/8k tested for both |
| **Transparent config** | All settings documented and version-controlled |
| **Reproducible** | Full source code + datasets + docker-compose |

## How to Run

### Prerequisites

```bash
# Install dependencies
pip install sochdb zep-python openai tiktoken numpy

# Set environment variables
export AZURE_OPENAI_API_KEY="your_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
export SOCHDB_LIB_PATH="/path/to/libsochdb_index.so"

# Optional: For Zep comparison
export ZEP_API_URL="http://localhost:8000"  # or Zep Cloud
export ZEP_API_KEY="your_zep_key"           # for Zep Cloud
```

### Run Benchmark

```bash
# Full benchmark suite (all 4 phases)
python3 benchmarks/run_memory_comparison.py

# Results saved to:
# benchmark_results/comparison_results_TIMESTAMP.json
```

### Example Output

```
======================================================================
  MEMORY SYSTEM COMPARISON BENCHMARK
  SochDB vs Zep
======================================================================

  Systems under test: SochDB, Zep
  Configuration:
    Token budgets: [2000, 4000, 8000]

======================================================================
  PHASE 1: MICROBENCHMARKS
======================================================================

  [SochDB]
    Generating test data... done
    Testing ingest... done
    Testing retrieval... done
    Results:
      Ingest (avg):     94.20ms
      Retrieval p50:    79.49ms
      Retrieval p95:    172.64ms
      Retrieval p99:    2557.91ms
      Context (avg):    36 tokens

  [Zep]
    ... (similar output)

======================================================================
  FINAL SUMMARY
======================================================================

  Phase 1: Microbenchmarks (Retrieval Latency)
  System       p50 (ms)   p95 (ms)   p99 (ms)
  ----------------------------------------------
  SochDB       79.49      172.64     2557.91
  Zep          85.23      145.67     1234.56

  Phase 2: Token Efficiency (2k budget)
  System       Avg Tokens Truncation
  ------------------------------------
  SochDB       1847       2.3%
  Zep          1923       1.8%

  ... (additional phases)
```

## Interpreting Results

### What "Good" Looks Like

**Retrieval Latency**:
- p50 < 100ms: Excellent (real-time UX)
- p95 < 200ms: Good (most users happy)
- p99 < 500ms: Acceptable (rare bad cases)

**Token Efficiency**:
- Budget utilization: 80-95% is ideal
- Truncation rate: <5% is good
- Lower tokens = lower API costs

**Quality**:
- QA accuracy: >80% is good, >90% is excellent
- Consistency: Should not degrade over time

**Scaling**:
- Degradation: <10x from 100→2000 is excellent
- O(log n): SochDB HNSW should achieve this
- O(n): Brute-force degrades 20-50x

## Next Steps

### To Add More Systems

1. Implement `MemorySystemAdapter` interface
2. Add adapter to `adapters/`
3. Register in `run_memory_comparison.py`
4. Run benchmarks
5. Compare results

### To Run Custom Workloads

1. Edit `workload_generator.py`
2. Add your conversation templates
3. Adjust `num_users`, `sessions_per_user`, etc.
4. Re-run benchmarks

### To Validate Results

1. Check `benchmark_results/*.json`
2. Verify configuration matches expectations
3. Compare across multiple runs
4. Share results with community

## References

- **Zep vs Mem0 Blog**: https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
- **LoCoMo Benchmark**: https://github.com/snap-research/LoCoMo
- **SochDB Docs**: https://docs.rs/sochdb
- **Zep Docs**: https://docs.getzep.com
- **Our Previous Benchmarks**: See `BENCHMARK_SUMMARY.md` for O(n) vs O(log n) results

## License

MIT License - Open source, fully transparent, reproducible benchmarks.

---

**Built by**: Claude Code
**Date**: 2026-01-04
**Purpose**: Fair, rigorous comparison of agent memory systems
**Status**: ✅ Production-ready framework, ready for testing
