#!/usr/bin/env python3
"""
Quick ToonDB Memory Comparison
===============================

Fast benchmark focusing on key metrics with smaller dataset.

Usage:
    export TOONDB_LIB_PATH=/path/to/libtoondb_index.so
    python3 benchmarks/quick_comparison.py
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from memory_benchmark_harness import BenchmarkConfig
from adapters.toondb_adapter import ToonDBAdapter
from workload_generator import SyntheticWorkloadGenerator, QueryGenerator


def quick_benchmark():
    """Run quick benchmark with small dataset"""
    print("="*70)
    print("  QUICK MEMORY SYSTEM BENCHMARK - ToonDB")
    print("="*70)

    # Small config for speed
    config = {
        "db_path": "/tmp/toondb_quick_bench.db",
        "embedding_dim": 1536,
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "top_k": 10,
    }

    # Initialize adapter
    print("\n✓ Initializing ToonDB adapter...")
    adapter = ToonDBAdapter(config)
    adapter.reset()

    # Phase 1: Microbenchmark
    print("\n" + "="*70)
    print("  PHASE 1: INGEST & RETRIEVAL PERFORMANCE")
    print("="*70)

    print("\n  Generating test data (5 users × 2 sessions × 20 messages)...")
    conversations = SyntheticWorkloadGenerator.generate_users_and_sessions(
        num_users=5,
        sessions_per_user=2,
        messages_per_session=20
    )
    print(f"  ✓ Generated {len(conversations)} conversations")

    # Test ingest
    print("\n  Testing ingest performance...")
    ingest_latencies = []

    for user_id, session_id, messages in conversations:
        latency = adapter.ingest_messages(user_id, session_id, messages)
        ingest_latencies.append(latency)

    avg_ingest = statistics.mean(ingest_latencies)
    print(f"  ✓ Ingest complete")
    print(f"    Avg latency: {avg_ingest:.2f}ms per conversation ({len(conversations[0][2])} msgs)")

    # Test retrieval
    print("\n  Testing retrieval performance (50 queries)...")
    queries = QueryGenerator.generate_queries(50)

    retrieval_latencies = []
    context_tokens = []

    for query in queries:
        user_id, session_id, _ = conversations[0]

        result = adapter.retrieve_context(
            user_id, session_id, query, token_budget=2000
        )

        retrieval_latencies.append(result.retrieval_latency_ms)
        context_tokens.append(result.token_count)

    sorted_lat = sorted(retrieval_latencies)
    n = len(sorted_lat)

    p50 = sorted_lat[n // 2]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]
    avg_tokens = statistics.mean(context_tokens) if context_tokens else 0

    print(f"  ✓ Retrieval complete")
    print(f"    p50 latency: {p50:.2f}ms")
    print(f"    p95 latency: {p95:.2f}ms")
    print(f"    p99 latency: {p99:.2f}ms")
    print(f"    Avg context: {avg_tokens:.0f} tokens")

    # Phase 2: Scale Test
    print("\n" + "="*70)
    print("  PHASE 2: SCALING BEHAVIOR")
    print("="*70)

    scales = [50, 100, 200]
    scale_results = {}

    for scale in scales:
        print(f"\n  Testing at {scale} observations...")
        adapter.reset()

        # Generate scaled data
        num_sessions = scale // 20  # 20 messages per session
        convs = SyntheticWorkloadGenerator.generate_users_and_sessions(
            num_users=5,
            sessions_per_user=num_sessions,
            messages_per_session=20
        )

        # Ingest
        for user_id, session_id, messages in convs:
            adapter.ingest_messages(user_id, session_id, messages)

        # Test retrieval
        test_queries = QueryGenerator.generate_queries(20)
        latencies = []

        for query in test_queries:
            user_id, session_id, _ = convs[0]
            result = adapter.retrieve_context(user_id, session_id, query, 2000)
            latencies.append(result.retrieval_latency_ms)

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        p95_lat = sorted_lat[int(n * 0.95)]

        scale_results[scale] = p95_lat
        print(f"    p95 latency: {p95_lat:.2f}ms")

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    print("\n  Performance Metrics:")
    print(f"    Ingest (avg):      {avg_ingest:.2f}ms")
    print(f"    Retrieval p50:     {p50:.2f}ms")
    print(f"    Retrieval p95:     {p95:.2f}ms")
    print(f"    Retrieval p99:     {p99:.2f}ms")
    print(f"    Context tokens:    {avg_tokens:.0f}")

    print("\n  Scaling (p95 latency):")
    for scale, lat in scale_results.items():
        print(f"    {scale:3d} obs: {lat:6.2f}ms")

    if len(scale_results) >= 2:
        degradation = scale_results[max(scale_results.keys())] / scale_results[min(scale_results.keys())]
        print(f"\n  Degradation: {degradation:.1f}x from {min(scale_results.keys())} → {max(scale_results.keys())} obs")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "ingest_avg_ms": avg_ingest,
            "retrieval_p50_ms": p50,
            "retrieval_p95_ms": p95,
            "retrieval_p99_ms": p99,
            "avg_context_tokens": avg_tokens,
        },
        "scaling": scale_results,
    }

    output_file = "benchmark_results/quick_comparison_results.json"
    os.makedirs("benchmark_results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("\n" + "="*70)

    adapter.close()
    return results


if __name__ == "__main__":
    # Check prerequisites
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY not set")
        sys.exit(1)

    quick_benchmark()
