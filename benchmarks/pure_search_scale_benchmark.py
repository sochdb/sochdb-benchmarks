#!/usr/bin/env python3
"""
Pure Vector Search Scaling Benchmark: O(n) vs O(log n)

This benchmark pre-generates embeddings to isolate pure vector search performance,
demonstrating the exact scaling behavior you described:

Problem: Brute-force O(n) causes P99 latency to degrade 143ms → 7.25s (50x) at scale
Solution: HNSW O(log n) keeps P99 latency ~50-100ms regardless of scale

Usage:
    python3 benchmarks/pure_search_scale_benchmark.py
"""

import time
import json
import statistics
from typing import List, Tuple
import numpy as np


# =============================================================================
# Pure Vector Search Implementations (No LLM calls during search)
# =============================================================================

class BruteForceSearch:
    """Brute-force O(n) vector search"""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []

    def add(self, text: str, embedding: np.ndarray):
        """Add vector - O(1)"""
        self.vectors.append(embedding)
        self.texts.append(text)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], float]:
        """Search - O(n) brute-force scan"""
        start = time.perf_counter()

        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # O(n): Compute similarity with ALL vectors
        similarities = []
        for i, vec in enumerate(self.vectors):
            vec_norm = vec / np.linalg.norm(vec)
            similarity = float(np.dot(query_norm, vec_norm))
            similarities.append((i, similarity))

        # Sort and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [self.texts[i] for i, _ in similarities[:k]]

        latency = time.perf_counter() - start
        return results, latency


class HNSWSearch:
    """HNSW O(log n) vector search"""

    def __init__(self, dimension: int = 1536):
        from sochdb import VectorIndex

        self.dimension = dimension
        self.index = VectorIndex(
            dimension=dimension,
            max_connections=16,
            ef_construction=100
        )
        self.texts: List[str] = []

    def add(self, text: str, embedding: np.ndarray):
        """Add vector - O(log n) with HNSW"""
        idx = len(self.texts)
        self.index.insert_batch(
            np.array([idx], dtype=np.uint64),
            np.array([embedding], dtype=np.float32)
        )
        self.texts.append(text)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], float]:
        """Search - O(log n) HNSW"""
        start = time.perf_counter()

        # O(log n): HNSW approximate nearest neighbor
        results = self.index.search(query_embedding, k=k)

        texts = [self.texts[int(idx)] for idx, _ in results if int(idx) < len(self.texts)]

        latency = time.perf_counter() - start
        return texts, latency


# =============================================================================
# Benchmark Runner
# =============================================================================

def generate_random_embeddings(count: int, dimension: int = 1536) -> List[np.ndarray]:
    """Generate random embeddings for testing"""
    return [np.random.randn(dimension).astype(np.float32) for _ in range(count)]


def run_scale_test(scale: int, num_queries: int = 100) -> dict:
    """Run benchmark at a specific scale"""
    print(f"\n{'='*70}")
    print(f"  SCALE: {scale} observations")
    print(f"{'='*70}")

    # Generate test data
    print(f"  Generating {scale} random embeddings...", end="", flush=True)
    embeddings = generate_random_embeddings(scale)
    query_embeddings = generate_random_embeddings(num_queries)
    texts = [f"observation_{i}" for i in range(scale)]
    print(" done")

    results = {}

    # Test Brute-Force
    print(f"\n  [Brute-Force O(n)]")
    bf = BruteForceSearch()

    print(f"    Loading {scale} vectors...", end="", flush=True)
    load_start = time.time()
    for text, emb in zip(texts, embeddings):
        bf.add(text, emb)
    load_time = time.time() - load_start
    print(f" done ({load_time:.2f}s)")

    print(f"    Running {num_queries} searches...", end="", flush=True)
    bf_latencies = []
    for query_emb in query_embeddings:
        _, latency = bf.search(query_emb, k=5)
        bf_latencies.append(latency * 1000)  # Convert to ms
    print(f" done")

    sorted_bf = sorted(bf_latencies)
    n = len(sorted_bf)
    results["brute_force"] = {
        "p50_ms": sorted_bf[n // 2],
        "p95_ms": sorted_bf[int(n * 0.95)],
        "p99_ms": sorted_bf[int(n * 0.99)],
        "avg_ms": statistics.mean(bf_latencies),
    }

    # Test HNSW
    print(f"\n  [HNSW O(log n)]")
    hnsw = HNSWSearch()

    print(f"    Loading {scale} vectors...", end="", flush=True)
    load_start = time.time()
    for text, emb in zip(texts, embeddings):
        hnsw.add(text, emb)
    load_time = time.time() - load_start
    print(f" done ({load_time:.2f}s)")

    print(f"    Running {num_queries} searches...", end="", flush=True)
    hnsw_latencies = []
    for query_emb in query_embeddings:
        _, latency = hnsw.search(query_emb, k=5)
        hnsw_latencies.append(latency * 1000)  # Convert to ms
    print(f" done")

    sorted_hnsw = sorted(hnsw_latencies)
    results["hnsw"] = {
        "p50_ms": sorted_hnsw[n // 2],
        "p95_ms": sorted_hnsw[int(n * 0.95)],
        "p99_ms": sorted_hnsw[int(n * 0.99)],
        "avg_ms": statistics.mean(hnsw_latencies),
    }

    # Compare
    speedup_p99 = results["brute_force"]["p99_ms"] / results["hnsw"]["p99_ms"]
    print(f"\n  Results:")
    print(f"    Brute-Force P99: {results['brute_force']['p99_ms']:.2f}ms")
    print(f"    HNSW P99:        {results['hnsw']['p99_ms']:.2f}ms")
    print(f"    Speedup:         {speedup_p99:.1f}x FASTER with HNSW")

    return {
        "scale": scale,
        "brute_force": results["brute_force"],
        "hnsw": results["hnsw"],
        "speedup_p99": speedup_p99,
    }


def main():
    """Run pure search scaling benchmark"""
    print("="*70)
    print("  PURE VECTOR SEARCH SCALING BENCHMARK")
    print("  Brute-Force O(n) vs HNSW O(log n)")
    print("="*70)
    print("\n  No LLM calls - pure vector search performance only")
    print("  This isolates the O(n) vs O(log n) scaling behavior")

    # Test at different scales
    scales = [40, 100, 200, 500, 1000, 2000]
    all_results = []

    for scale in scales:
        result = run_scale_test(scale, num_queries=100)
        all_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    print("\n  Brute-Force O(n) Search:")
    print(f"  {'Scale':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("  " + "-"*48)
    for r in all_results:
        bf = r["brute_force"]
        print(f"  {r['scale']:<10} {bf['p50_ms']:<12.3f} {bf['p95_ms']:<12.3f} {bf['p99_ms']:<12.3f}")

    print("\n  HNSW O(log n) Search:")
    print(f"  {'Scale':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("  " + "-"*48)
    for r in all_results:
        hnsw = r["hnsw"]
        print(f"  {r['scale']:<10} {hnsw['p50_ms']:<12.3f} {hnsw['p95_ms']:<12.3f} {hnsw['p99_ms']:<12.3f}")

    print("\n  Speedup (HNSW vs Brute-Force):")
    print(f"  {'Scale':<10} {'P99 Speedup':<15}")
    print("  " + "-"*25)
    for r in all_results:
        print(f"  {r['scale']:<10} {r['speedup_p99']:<15.1f}x")

    # Analysis
    if len(all_results) >= 3:
        small = all_results[0]  # 40
        medium = all_results[2]  # 200
        large = all_results[-1]  # 2000

        bf_degradation = large["brute_force"]["p99_ms"] / small["brute_force"]["p99_ms"]
        hnsw_degradation = large["hnsw"]["p99_ms"] / small["hnsw"]["p99_ms"]

        print("\n" + "="*70)
        print("  SCALING ANALYSIS")
        print("="*70)

        print(f"\n  Brute-Force ({small['scale']} → {large['scale']} observations):")
        print(f"    P99: {small['brute_force']['p99_ms']:.2f}ms → {large['brute_force']['p99_ms']:.2f}ms")
        print(f"    Degradation: {bf_degradation:.1f}x WORSE")

        print(f"\n  HNSW ({small['scale']} → {large['scale']} observations):")
        print(f"    P99: {small['hnsw']['p99_ms']:.2f}ms → {large['hnsw']['p99_ms']:.2f}ms")
        print(f"    Degradation: {hnsw_degradation:.1f}x")

        print(f"\n  🎯 KEY FINDING:")
        print(f"    At {medium['scale']} observations: HNSW is {medium['speedup_p99']:.1f}x faster")
        print(f"    At {large['scale']} observations: HNSW is {large['speedup_p99']:.1f}x faster")
        print(f"    Brute-force degrades {bf_degradation:.0f}x, HNSW only {hnsw_degradation:.1f}x")

    # Save results
    output_file = f"pure_search_scale_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
