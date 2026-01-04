#!/usr/bin/env python3
"""
Vector Workload Runner for ToonDB

Runs vector search benchmarks using actual ToonDB VectorIndex.
Measures insert throughput, search QPS, latency percentiles, and recall@k.
"""

import argparse
import json
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

# HDRHistogram-like percentile calculation
class LatencyHistogram:
    """Simple latency histogram for percentile calculation."""
    
    def __init__(self):
        self.latencies_ns: List[int] = []
    
    def record_ns(self, latency_ns: int):
        self.latencies_ns.append(latency_ns)
    
    def record_s(self, latency_s: float):
        self.latencies_ns.append(int(latency_s * 1e9))
    
    def percentile(self, p: float) -> float:
        """Get percentile in milliseconds."""
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
    
    def count(self) -> int:
        return len(self.latencies_ns)


def load_dataset(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Load dataset from directory."""
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset meta.json not found: {meta_path}")
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    dimension = meta["dimension"]
    
    # Load embeddings
    embeddings_path = dataset_dir / meta.get("files", {}).get("embeddings", "embeddings.f32")
    if embeddings_path.exists():
        embeddings = np.fromfile(embeddings_path, dtype=np.float32)
        num_vectors = len(embeddings) // dimension
        embeddings = embeddings.reshape(num_vectors, dimension)
    else:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    # Load queries
    queries_path = dataset_dir / meta.get("files", {}).get("queries", "queries.f32")
    if queries_path.exists():
        queries = np.fromfile(queries_path, dtype=np.float32)
        num_queries = len(queries) // dimension
        queries = queries.reshape(num_queries, dimension)
    else:
        # Generate random queries if not available
        queries = np.random.randn(100, dimension).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Load ground truth
    truth_path = dataset_dir / meta.get("files", {}).get("ground_truth", "ground_truth.json")
    ground_truth = None
    if truth_path.exists():
        with open(truth_path) as f:
            ground_truth = json.load(f)
    
    return embeddings, queries, ground_truth


def compute_recall(results: List[int], truth: List[int], k: int) -> float:
    """Compute recall@k."""
    if not truth:
        return 1.0
    retrieved = set(results[:k])
    relevant = set(truth[:k])
    if not relevant:
        return 1.0
    return len(retrieved & relevant) / len(relevant)


def run_vector_benchmark(
    embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: Optional[dict],
    config: dict
) -> dict:
    """Run the vector benchmark using ToonDB VectorIndex."""
    
    # Import ToonDB
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "toondb-python-sdk/src"))
        from toondb import VectorIndex
        if VectorIndex is None:
            raise ImportError("VectorIndex not available")
    except ImportError as e:
        print(f"Error: Could not import ToonDB VectorIndex: {e}")
        print("Make sure TOONDB_LIB_PATH is set to the directory containing libtoondb_index")
        sys.exit(1)
    
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    num_queries = queries.shape[0]
    
    # Configuration
    M = config.get("M", 16)
    ef_construction = config.get("ef_construction", 100)
    ef_search = config.get("ef_search", 64)
    k = config.get("k", 10)
    batch_size = config.get("batch_size", 1000)
    
    print(f"\n=== ToonDB Vector Benchmark ===")
    print(f"Vectors: {num_vectors:,} @ {dimension}-dim")
    print(f"Queries: {num_queries:,}")
    print(f"HNSW: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
    print(f"k={k}, batch_size={batch_size}")
    print()
    
    # Create index
    print("Creating index...")
    index = VectorIndex(
        dimension=dimension,
        max_connections=M,
        ef_construction=ef_construction,
    )
    
    # Insert with batching
    print("Inserting vectors...")
    insert_histogram = LatencyHistogram()
    insert_start = time.perf_counter()
    
    for batch_start in range(0, num_vectors, batch_size):
        batch_end = min(batch_start + batch_size, num_vectors)
        ids = np.arange(batch_start, batch_end, dtype=np.uint64)
        vectors = embeddings[batch_start:batch_end]
        
        batch_start_time = time.perf_counter()
        inserted = index.insert_batch(ids, vectors)
        batch_elapsed = time.perf_counter() - batch_start_time
        
        insert_histogram.record_s(batch_elapsed / len(ids))
        
        if (batch_end % 10000) == 0 or batch_end == num_vectors:
            elapsed = time.perf_counter() - insert_start
            rate = batch_end / elapsed
            print(f"  Inserted {batch_end:,}/{num_vectors:,} ({rate:,.0f} vec/s)")
    
    insert_total = time.perf_counter() - insert_start
    insert_rate = num_vectors / insert_total
    
    print(f"\nInsert complete: {insert_rate:,.0f} vec/s ({insert_total:.2f}s)")
    
    # Search with latency measurement
    print("\nRunning search queries...")
    search_histogram = LatencyHistogram()
    recalls: List[float] = []
    
    search_start = time.perf_counter()
    
    for i, query in enumerate(queries):
        query_start = time.perf_counter()
        results = index.search(query, k=k)
        query_elapsed = time.perf_counter() - query_start
        
        search_histogram.record_s(query_elapsed)
        
        # Compute recall if ground truth available
        if ground_truth and "queries" in ground_truth:
            query_truth = ground_truth["queries"].get(str(i), {})
            true_neighbors = query_truth.get("neighbors", [])
            result_ids = [r[0] for r in results]
            recall = compute_recall(result_ids, true_neighbors, k)
            recalls.append(recall)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - search_start
            qps = (i + 1) / elapsed
            print(f"  Searched {i+1:,}/{num_queries:,} ({qps:,.0f} QPS)")
    
    search_total = time.perf_counter() - search_start
    qps = num_queries / search_total
    
    print(f"\nSearch complete: {qps:,.0f} QPS ({search_total:.2f}s)")
    
    # Compile results
    results = {
        "workload": "toondb_vector",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_vectors": num_vectors,
            "num_queries": num_queries,
            "dimension": dimension,
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "k": k,
        },
        "insert": {
            "total_s": insert_total,
            "rate_vec_per_s": insert_rate,
            "p50_ms": insert_histogram.percentile(50),
            "p95_ms": insert_histogram.percentile(95),
            "p99_ms": insert_histogram.percentile(99),
        },
        "search": {
            "total_s": search_total,
            "qps": qps,
            "p50_ms": search_histogram.percentile(50),
            "p95_ms": search_histogram.percentile(95),
            "p99_ms": search_histogram.percentile(99),
            "p999_ms": search_histogram.percentile(99.9),
            "mean_ms": search_histogram.mean_ms(),
        },
    }
    
    if recalls:
        results["quality"] = {
            f"recall@{k}": {
                "mean": sum(recalls) / len(recalls),
                "min": min(recalls),
                "max": max(recalls),
            }
        }
        print(f"\nRecall@{k}: {results['quality'][f'recall@{k}']['mean']:.4f}")
    
    return results


def print_summary(results: dict):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    
    insert = results["insert"]
    search = results["search"]
    config = results["config"]
    
    print(f"\nDataset: {config['num_vectors']:,} vectors × {config['dimension']}-dim")
    print(f"Queries: {config['num_queries']:,}")
    
    print(f"\nInsert Performance:")
    print(f"  Rate:     {insert['rate_vec_per_s']:,.0f} vec/s")
    print(f"  p50:      {insert['p50_ms']:.3f} ms")
    print(f"  p99:      {insert['p99_ms']:.3f} ms")
    
    print(f"\nSearch Performance:")
    print(f"  QPS:      {search['qps']:,.0f}")
    print(f"  p50:      {search['p50_ms']:.3f} ms")
    print(f"  p95:      {search['p95_ms']:.3f} ms")
    print(f"  p99:      {search['p99_ms']:.3f} ms")
    print(f"  p99.9:    {search['p999_ms']:.3f} ms")
    
    if "quality" in results:
        k = config["k"]
        quality = results["quality"][f"recall@{k}"]
        print(f"\nQuality:")
        print(f"  Recall@{k}: {quality['mean']:.4f} (min: {quality['min']:.4f}, max: {quality['max']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="ToonDB Vector Workload Runner")
    parser.add_argument("--dataset", "-d", help="Dataset directory")
    parser.add_argument("--vectors", "-n", type=int, default=10000, help="Number of vectors (if no dataset)")
    parser.add_argument("--dim", type=int, default=128, help="Dimension (if no dataset)")
    parser.add_argument("--queries", "-q", type=int, default=1000, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=100, help="ef_construction")
    parser.add_argument("--ef-search", type=int, default=64, help="ef_search")
    parser.add_argument("--batch-size", type=int, default=1000, help="Insert batch size")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--json", action="store_true", help="Output only JSON")
    
    args = parser.parse_args()
    
    config = {
        "M": args.M,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "k": args.k,
        "batch_size": args.batch_size,
    }
    
    if args.dataset:
        # Load from dataset
        dataset_dir = Path(args.dataset)
        if not args.json:
            print(f"Loading dataset from {dataset_dir}...")
        embeddings, queries, ground_truth = load_dataset(dataset_dir)
        if args.queries < len(queries):
            queries = queries[:args.queries]
    else:
        # Generate synthetic data
        if not args.json:
            print(f"Generating synthetic data: {args.vectors} vectors × {args.dim}-dim")
        embeddings = np.random.randn(args.vectors, args.dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        queries = np.random.randn(args.queries, args.dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        ground_truth = None
    
    # Run benchmark
    results = run_vector_benchmark(embeddings, queries, ground_truth, config)
    
    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.json:
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
