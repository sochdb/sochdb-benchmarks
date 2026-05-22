# LoCoMo Retrieval-Only Benchmark: SochDB-backed NVIDIA Hybrid Baseline

## Summary

This report records a LoCoMo retrieval-only benchmark for a SochDB-backed hybrid retrieval pipeline.

The benchmark evaluates evidence recovery, not answer-generation accuracy.

## Valid Claim

This benchmark shows that SochDB can serve as the dense vector backend inside a hybrid sparse+dense memory retrieval pipeline.

## Scope

This is not a pure SochDB-only benchmark. The pipeline uses:

- BM25 for sparse lexical retrieval.
- NVIDIA `nvidia/llama-nemotron-embed-1b-v2` for dense embeddings.
- SochDB gRPC as the dense vector retrieval backend.
- Reciprocal Rank Fusion for combining sparse and dense candidates.

Answer-generation and LLM judge accuracy are intentionally excluded from this report.

## Benchmark Configuration

| Field | Value |
|---|---|
| Benchmark | LoCoMo retrieval-only |
| Dataset | LoCoMo converted QA split |
| Questions | 1,986 |
| Memory rows | 5,882 raw conversation turns |
| Sparse retriever | BM25 |
| Dense embedding model | `nvidia/llama-nemotron-embed-1b-v2` |
| Dense backend | SochDB gRPC |
| Fusion | RRF |
| `k` | 20 |
| `candidate_k` | 100 |
| `bm25_weight` | 1.5 |
| `vector_weight` | 0.75 |

## Result

| System | Questions | Evidence Hit@20 | Evidence Recall@20 | Avg Context Tokens | Avg Latency |
|---|---:|---:|---:|---:|---:|
| BM25 + SochDB vector + NVIDIA Nemotron + RRF | 1,986 | 0.7056 | 0.6522 | 657.98 | 200.81 ms |

## Category Breakdown

| Category | Questions | Evidence Hit@20 | Evidence Recall@20 | Avg Context Tokens | Avg Latency |
|---|---:|---:|---:|---:|---:|
| adversarial | 446 | 0.7534 | 0.7455 | 668.59 | 200.87 ms |
| multi_hop | 96 | 0.4157 | 0.3058 | 646.21 | 200.85 ms |
| open_domain | 841 | 0.7337 | 0.7210 | 656.27 | 200.90 ms |
| single_hop | 282 | 0.5801 | 0.3271 | 650.08 | 200.66 ms |
| temporal | 321 | 0.7563 | 0.7232 | 658.21 | 200.61 ms |

## Interpretation

The strongest retrieval performance appears in temporal, adversarial, and open-domain questions.

The weakest categories are multi-hop and single-hop. Multi-hop likely needs graph/fact expansion or structured memory traversal. Single-hop likely needs better exact fact extraction and better handling of short evidence spans.

## Pipeline

```text
LoCoMo raw conversation turns
        ↓
Converted memory rows + question rows
        ↓
BM25 sparse retrieval top-100
        +
NVIDIA Nemotron embeddings
        ↓
SochDB gRPC dense vector retrieval top-100
        ↓
RRF fusion
        ↓
Top-20 retrieved memories
        ↓
Compare retrieved memory IDs against gold evidence memory IDs
        ↓
Evidence Hit@20 / Evidence Recall@20
