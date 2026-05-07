#!/usr/bin/env python3
"""
Bulk CLI vector workload runner for SochDB.

This path is intended for large staged benchmarks on machines where the bulk CLI
is available but the in-process Python native extension path is not reliable.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from sochdb import build_index_from_file
from sochdb._bulk import bulk_query_from_file


class LatencyHistogram:
    def __init__(self) -> None:
        self.latencies_ns: List[int] = []

    def record_s(self, latency_s: float) -> None:
        self.latencies_ns.append(int(latency_s * 1e9))

    def percentile(self, p: float) -> float:
        if not self.latencies_ns:
            return 0.0
        sorted_lat = sorted(self.latencies_ns)
        idx = int(len(sorted_lat) * p / 100)
        idx = min(idx, len(sorted_lat) - 1)
        return sorted_lat[idx] / 1e6

    def mean_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        return sum(self.latencies_ns) / len(self.latencies_ns) / 1e6


def load_dataset_meta(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="SochDB bulk CLI vector workload runner")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    parser.add_argument("--queries", type=int, default=250, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=100, help="ef_construction")
    parser.add_argument("--ef-search", type=int, default=64, help="ef_search")
    parser.add_argument("--batch-size", type=int, default=1000, help="Build batch size")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--index-path", required=True, help="Output path for built index")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    meta = load_dataset_meta(dataset_dir)
    dim = int(meta["dimension"])
    num_vectors = int(meta["num_vectors"])
    num_queries = min(args.queries, int(meta["num_queries"]))

    embeddings_path = dataset_dir / meta.get("files", {}).get("embeddings", "embeddings.f32")
    queries_path = dataset_dir / meta.get("files", {}).get("queries", "queries.f32")
    all_queries = np.fromfile(queries_path, dtype=np.float32).reshape(-1, dim)[:num_queries]

    print(f"\n=== SochDB Bulk Vector Benchmark ===")
    print(f"Vectors: {num_vectors:,} @ {dim}-dim")
    print(f"Queries: {num_queries:,}")
    print(f"HNSW: M={args.M}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    print(f"k={args.k}, batch_size={args.batch_size}")
    print()

    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building index from file...")
    build_start = time.perf_counter()
    build_stats = build_index_from_file(
        input_path=str(embeddings_path),
        output_path=str(index_path),
        dimension=dim,
        m=args.M,
        ef_construction=args.ef_construction,
        batch_size=args.batch_size,
        quiet=False,
    )
    build_total = time.perf_counter() - build_start

    if not build_stats.get("vectors"):
        build_stats["vectors"] = num_vectors
        build_stats["rate"] = num_vectors / build_total if build_total > 0 else 0.0

    print(f"Build complete: {build_stats['rate']:,.0f} vec/s ({build_total:.2f}s)")

    search_hist = LatencyHistogram()
    tmp_dir = Path(tempfile.mkdtemp(prefix="sochdb-bulk-query-"))

    print("\nRunning CLI search queries...")
    search_start = time.perf_counter()
    for i, query in enumerate(all_queries):
        query_file = tmp_dir / f"q_{i}.f32"
        query.astype(np.float32).tofile(query_file)

        query_start = time.perf_counter()
        _ = bulk_query_from_file(
            index_path=str(index_path),
            query_path=str(query_file),
            k=args.k,
            ef_search=args.ef_search,
        )
        query_elapsed = time.perf_counter() - query_start
        search_hist.record_s(query_elapsed)

        query_file.unlink(missing_ok=True)
        if (i + 1) % 50 == 0 or (i + 1) == num_queries:
            elapsed = time.perf_counter() - search_start
            qps = (i + 1) / elapsed
            print(f"  Searched {i + 1:,}/{num_queries:,} ({qps:,.1f} QPS)")

    search_total = time.perf_counter() - search_start
    qps = num_queries / search_total if search_total > 0 else 0.0

    results = {
        "workload": "sochdb_bulk_cli_vector",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_vectors": num_vectors,
            "num_queries": num_queries,
            "dimension": dim,
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "k": args.k,
            "batch_size": args.batch_size,
            "index_path": str(index_path),
        },
        "build": {
            "total_s": build_total,
            "rate_vec_per_s": build_stats.get("rate", 0.0),
            "output_size_mb": build_stats.get("output_size_mb", 0.0),
        },
        "search": {
            "total_s": search_total,
            "qps": qps,
            "p50_ms": search_hist.percentile(50),
            "p95_ms": search_hist.percentile(95),
            "p99_ms": search_hist.percentile(99),
            "mean_ms": search_hist.mean_ms(),
        },
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
