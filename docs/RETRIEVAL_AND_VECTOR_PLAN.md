# SochDB Benchmarks: Retrieval and Vector Plan

## Why this exists

This repo already contains a lot of benchmark scripts and reports, but the current surface is broad and somewhat scattered.

For ML engineering and product evaluation, we need a clearer benchmark story built around a few strong tracks instead of many disconnected files.

## Benchmark tracks we should emphasize

### 1. Retrieval Quality

Use datasets like:

- SciFact
- later other BEIR-style datasets if needed

Measure:

- recall@k
- MRR
- nDCG
- latency

This track answers:
- does SochDB actually retrieve useful results?

Quality-first rule:

- keep embeddings fixed while tuning the index
- compare embeddings separately from ANN settings

### 2. Vector Engine / ANN Performance

Use datasets like:

- `glove100_angular`

Measure:

- recall vs `ef_search`
- latency vs `ef_search`
- insert throughput
- search throughput

This track answers:
- how strong is the vector engine itself?

### 3. Product-Facing Comparisons

Compare SochDB to:

- Qdrant
- pgvector
- ChromaDB
- LanceDB

But keep the comparisons clearly labeled:

- measured by us
- claimed from vendor / blog / PR
- reproduced locally

## Recommended structure to converge toward

```text
sochdb-benchmarks/
  benchmarks/
    retrieval/
    ann/
    memory/
  datasets/
    retrieval/
    ann/
  reports/
    retrieval/
    ann/
    comparisons/
  docs/
    methodology/
```

This does not require a large refactor immediately, but new benchmark work should aim toward this structure.

## Priority benchmark work

### B-1 SciFact reproducibility

Keep one clean path for:

- dataset prep
- embedding generation
- SochDB local run
- SochDB 2.0 remote/local run
- evaluation

The first quality pass should be:

- same SciFact embeddings
- HNSW sweep across `m`, `ef_construction`, and `ef_search`
- summary table of `recall@k`, `MRR`, `nDCG@k`, and `p95`

### B-2 `glove100_angular` track

Create one benchmark path focused on:

- ef_search sweeps
- recall / latency tradeoff
- apples-to-apples comparison against Qdrant when possible

### B-3 Comparison note

Maintain one concise comparison document that separates:

- measured by us
- observed live
- claimed by others

## How to think about “better quality”

There are three separate levers:

### 1. Better embeddings

Examples:

- stronger sentence-transformer model
- OpenAI or Azure embedding model
- domain-specific finetuned model

This usually has the biggest effect on retrieval relevance.

### 2. Better ANN tuning

Examples:

- higher `m`
- higher `ef_construction`
- higher `ef_search`

This improves recall against a fixed embedding space, usually at a latency and build-cost tradeoff.

### 3. Different index families

Examples:

- HNSW
- IVF / PQ
- exact search
- reranked hybrid pipelines

We should not jump here first. HNSW is already a strong default, and we should characterize it properly before replacing it.

## Recommended decision order

1. prove the HNSW quality envelope with sweeps
2. compare embedding models on the same sweep baseline
3. only then explore other ANN/index families if the quality-latency frontier still looks weak

## Studio integration

This repo should become the best source for Studio’s:

- benchmark cards
- evidence panel
- comparison charts
- reproducible benchmark metadata

That means reports should be shaped so `sochdb-studio` can consume them more easily later.
