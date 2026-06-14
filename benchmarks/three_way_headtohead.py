#!/usr/bin/env python3
"""
Full end-to-end head-to-head: SochDB vs ChromaDB vs LanceDB

Tests insert throughput, search latency (p50/p95/p99), QPS, and Recall@10
across multiple vector dimensions and scales.

Usage:
    python benchmarks/three_way_headtohead.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


TOP_K = 10
NUM_QUERIES = 100
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100


@dataclass
class BenchResult:
    system: str
    config: str
    num_vectors: int
    dimension: int
    insert_s: float
    insert_vec_per_s: float
    search_avg_ms: float
    search_p50_ms: float
    search_p95_ms: float
    search_p99_ms: float
    qps: float
    recall_at_10: float


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def brute_force_topk(vectors: np.ndarray, query: np.ndarray, k: int) -> List[int]:
    scores = vectors @ query
    return np.argsort(scores)[-k:][::-1].tolist()


def recall_at_k(approx: List[int], exact: List[int], k: int) -> float:
    return len(set(approx[:k]) & set(exact[:k])) / k


def latency_stats(latencies_ms: List[float]) -> Tuple[float, float, float, float]:
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    avg = sum(sorted_l) / n
    p50 = sorted_l[n // 2]
    p95 = sorted_l[int(n * 0.95)]
    p99 = sorted_l[int(n * 0.99)]
    return avg, p50, p95, p99


def benchmark_sochdb(
    vectors: np.ndarray, queries: np.ndarray, config: str
) -> BenchResult:
    from sochdb import VectorIndex

    dim = vectors.shape[1]
    index = VectorIndex(
        dimension=dim,
        max_connections=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
    )

    ids = np.arange(len(vectors), dtype=np.uint64)
    start = time.perf_counter()
    index.insert_batch(ids, vectors)
    insert_s = time.perf_counter() - start

    for q in queries[:5]:
        index.search(q, k=TOP_K)

    latencies = []
    recalls = []
    for query in queries:
        t0 = time.perf_counter()
        results = index.search(query, k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        approx = [int(r[0]) for r in results]
        exact = brute_force_topk(vectors, query, TOP_K)
        recalls.append(recall_at_k(approx, exact, TOP_K))

    avg, p50, p95, p99 = latency_stats(latencies)
    return BenchResult(
        system="SochDB",
        config=config,
        num_vectors=len(vectors),
        dimension=dim,
        insert_s=insert_s,
        insert_vec_per_s=len(vectors) / insert_s,
        search_avg_ms=avg,
        search_p50_ms=p50,
        search_p95_ms=p95,
        search_p99_ms=p99,
        qps=1000 / avg,
        recall_at_10=float(np.mean(recalls)),
    )


def benchmark_chromadb(
    vectors: np.ndarray, queries: np.ndarray, config: str
) -> BenchResult:
    import chromadb

    tmp_dir = tempfile.mkdtemp(prefix="chroma_h2h_")
    try:
        client = chromadb.PersistentClient(path=tmp_dir)
        collection = client.create_collection(
            name="benchmark",
            metadata={"hnsw:space": "cosine"},
        )

        ids = [str(i) for i in range(len(vectors))]
        embeddings = vectors.tolist()

        start = time.perf_counter()
        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            collection.add(ids=ids[i:end], embeddings=embeddings[i:end])
        insert_s = time.perf_counter() - start

        for q in queries[:5]:
            collection.query(query_embeddings=[q.tolist()], n_results=TOP_K)

        latencies = []
        recalls = []
        for query in queries:
            t0 = time.perf_counter()
            results = collection.query(
                query_embeddings=[query.tolist()], n_results=TOP_K
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            approx = [int(x) for x in results["ids"][0]]
            exact = brute_force_topk(vectors, query, TOP_K)
            recalls.append(recall_at_k(approx, exact, TOP_K))

        avg, p50, p95, p99 = latency_stats(latencies)
        return BenchResult(
            system="ChromaDB",
            config=config,
            num_vectors=len(vectors),
            dimension=vectors.shape[1],
            insert_s=insert_s,
            insert_vec_per_s=len(vectors) / insert_s,
            search_avg_ms=avg,
            search_p50_ms=p50,
            search_p95_ms=p95,
            search_p99_ms=p99,
            qps=1000 / avg,
            recall_at_10=float(np.mean(recalls)),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def benchmark_lancedb(
    vectors: np.ndarray, queries: np.ndarray, config: str
) -> BenchResult:
    import lancedb

    tmp_dir = tempfile.mkdtemp(prefix="lancedb_h2h_")
    try:
        db = lancedb.connect(tmp_dir)
        data = [{"id": i, "vector": vec.tolist()} for i, vec in enumerate(vectors)]

        start = time.perf_counter()
        table = db.create_table("vectors", data)
        insert_s = time.perf_counter() - start

        indexed = False
        dim = vectors.shape[1]
        for sub_dim in (96, 48, 32, 16, 8, 4, 2, 1):
            if dim % sub_dim != 0:
                continue
            try:
                num_partitions = min(256, max(4, len(vectors) // 39))
                table.create_index(
                    metric="cosine",
                    num_partitions=num_partitions,
                    num_sub_vectors=sub_dim,
                )
                indexed = True
                print(f"    LanceDB IVF-PQ index: partitions={num_partitions}, sub_vectors={sub_dim}")
                break
            except Exception as exc:
                print(f"    LanceDB index attempt (sub_vectors={sub_dim}) failed: {exc}")

        for q in queries[:5]:
            table.search(q.tolist()).limit(TOP_K).to_list()

        latencies = []
        recalls = []
        for query in queries:
            t0 = time.perf_counter()
            results = table.search(query.tolist()).limit(TOP_K).to_list()
            latencies.append((time.perf_counter() - t0) * 1000)
            approx = [int(r["id"]) for r in results]
            exact = brute_force_topk(vectors, query, TOP_K)
            recalls.append(recall_at_k(approx, exact, TOP_K))

        avg, p50, p95, p99 = latency_stats(latencies)
        result = BenchResult(
            system="LanceDB",
            config=config,
            num_vectors=len(vectors),
            dimension=vectors.shape[1],
            insert_s=insert_s,
            insert_vec_per_s=len(vectors) / insert_s,
            search_avg_ms=avg,
            search_p50_ms=p50,
            search_p95_ms=p95,
            search_p99_ms=p99,
            qps=1000 / avg,
            recall_at_10=float(np.mean(recalls)),
        )
        if not indexed:
            result.system = "LanceDB (no index)"
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


BENCHMARKS = [
    (10_000, 128),
    (10_000, 768),
    (10_000, 1536),
    (50_000, 128),
]

RUNNERS = [
    ("SochDB", benchmark_sochdb),
    ("ChromaDB", benchmark_chromadb),
    ("LanceDB", benchmark_lancedb),
]


def print_result(r: BenchResult):
    print(
        f"    Insert: {r.insert_s:.2f}s ({r.insert_vec_per_s:,.0f} vec/s) | "
        f"Search p50/p95/p99: {r.search_p50_ms:.2f}/{r.search_p95_ms:.2f}/{r.search_p99_ms:.2f}ms | "
        f"QPS: {r.qps:,.0f} | Recall@10: {r.recall_at_10:.1%}"
    )


def print_summary_table(results: List[BenchResult]):
    configs = sorted({r.config for r in results})
    systems = ["SochDB", "ChromaDB", "LanceDB"]

    print("\n" + "=" * 90)
    print("  INSERT THROUGHPUT (vec/s)")
    print("=" * 90)
    header = f"{'Config':<16}" + "".join(f"{s:>18}" for s in systems)
    print(header)
    print("-" * len(header))
    for cfg in configs:
        row = f"{cfg:<16}"
        for sys in systems:
            match = next((r for r in results if r.config == cfg and r.system.startswith(sys)), None)
            row += f"{match.insert_vec_per_s:>18,.0f}" if match else f"{'N/A':>18}"
        print(row)

    print("\n" + "=" * 90)
    print("  SEARCH LATENCY p50 (ms)")
    print("=" * 90)
    print(header)
    print("-" * len(header))
    for cfg in configs:
        row = f"{cfg:<16}"
        for sys in systems:
            match = next((r for r in results if r.config == cfg and r.system.startswith(sys)), None)
            row += f"{match.search_p50_ms:>18.3f}" if match else f"{'N/A':>18}"
        print(row)

    print("\n" + "=" * 90)
    print("  RECALL@10")
    print("=" * 90)
    print(header)
    print("-" * len(header))
    for cfg in configs:
        row = f"{cfg:<16}"
        for sys in systems:
            match = next((r for r in results if r.config == cfg and r.system.startswith(sys)), None)
            row += f"{match.recall_at_10:>17.1%}" if match else f"{'N/A':>18}"
        print(row)


def main():
    print("=" * 90)
    print("  FULL END-TO-END: SochDB vs ChromaDB vs LanceDB")
    print("=" * 90)
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Queries per config: {NUM_QUERIES} | Top-K: {TOP_K}")
    print(f"  HNSW params (SochDB/ChromaDB): M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}")

    all_results: List[BenchResult] = []

    for num_vectors, dim in BENCHMARKS:
        config = f"{num_vectors // 1000}K×{dim}"
        print(f"\n{'#' * 90}")
        print(f"  CONFIG: {num_vectors:,} vectors × {dim}-dim")
        print(f"{'#' * 90}")

        vectors = generate_vectors(num_vectors, dim, seed=42)
        queries = generate_vectors(NUM_QUERIES, dim, seed=123)

        for name, runner in RUNNERS:
            print(f"\n  [{name}]")
            try:
                result = runner(vectors, queries, config)
                print_result(result)
                all_results.append(result)
            except Exception as exc:
                print(f"    ERROR: {exc}")

    print_summary_table(all_results)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"three_way_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "num_queries": NUM_QUERIES,
            "top_k": TOP_K,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        },
        "results": [asdict(r) for r in all_results],
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()