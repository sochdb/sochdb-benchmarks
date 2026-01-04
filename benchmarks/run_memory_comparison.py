#!/usr/bin/env python3
"""
ToonDB vs Zep Memory System Comparison
=======================================

Apples-to-apples benchmark of agent memory systems.

Phases:
1. Microbenchmarks (ingest + retrieval latency)
2. Token efficiency (context assembly within budget)
3. LoCoMo quality (end-to-end QA accuracy)
4. Scale test (performance under load)

Usage:
    export TOONDB_LIB_PATH=/path/to/libtoondb_index.so
    export AZURE_OPENAI_API_KEY=your_key
    export AZURE_OPENAI_ENDPOINT=your_endpoint

    # Optional: For Zep comparison
    export ZEP_API_URL=http://localhost:8000  # or Zep Cloud URL
    export ZEP_API_KEY=your_zep_key

    python3 benchmarks/run_memory_comparison.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_benchmark_harness import (
    MemorySystemAdapter,
    BenchmarkConfig,
    BenchmarkMetrics,
    get_benchmark_config
)
from adapters.toondb_adapter import ToonDBAdapter
from adapters.zep_adapter import ZepAdapter
from workload_generator import (
    SyntheticWorkloadGenerator,
    LoCoMoDatasetLoader,
    QueryGenerator
)


# =============================================================================
# Benchmark Phases
# =============================================================================

class MemoryBenchmarkRunner:
    """Orchestrates benchmark phases"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        # Create results directory
        os.makedirs(config.results_dir, exist_ok=True)

        # Initialize adapters
        self.adapters: Dict[str, MemorySystemAdapter] = {}
        self._init_adapters()

    def _init_adapters(self):
        """Initialize memory system adapters"""
        # ToonDB
        toondb_config = {
            "db_path": "/tmp/toondb_benchmark.db",
            "embedding_dim": self.config.embedding_dim,
            "hnsw_m": 16,
            "hnsw_ef_construction": 100,
            "top_k": self.config.top_k,
        }

        try:
            self.adapters["ToonDB"] = ToonDBAdapter(toondb_config)
            print("✓ ToonDB adapter initialized")
        except Exception as e:
            print(f"✗ ToonDB initialization failed: {e}")

        # Zep
        zep_config = {
            "zep_api_url": os.getenv("ZEP_API_URL"),
            "zep_api_key": os.getenv("ZEP_API_KEY"),
            "top_k": self.config.top_k,
        }

        try:
            zep_adapter = ZepAdapter(zep_config)
            if zep_adapter.available:
                self.adapters["Zep"] = zep_adapter
                print("✓ Zep adapter initialized")
            else:
                print("✗ Zep adapter not available (skipping)")
        except Exception as e:
            print(f"✗ Zep initialization failed: {e}")

    def phase1_microbenchmarks(self):
        """
        Phase 1: Microbenchmarks

        Test pure ingest and retrieval performance
        """
        print("\n" + "="*70)
        print("  PHASE 1: MICROBENCHMARKS")
        print("="*70)

        results = {}

        for system_name, adapter in self.adapters.items():
            print(f"\n  [{system_name}]")
            metrics = BenchmarkMetrics()

            # Reset system
            adapter.reset()

            # Generate small test dataset
            print(f"    Generating test data...", end="", flush=True)
            conversations = SyntheticWorkloadGenerator.generate_users_and_sessions(
                num_users=10,
                sessions_per_user=5,
                messages_per_session=20
            )
            print(" done")

            # Test ingest performance
            print(f"    Testing ingest...", end="", flush=True)
            for user_id, session_id, messages in conversations:
                latency = adapter.ingest_messages(user_id, session_id, messages)
                metrics.add_ingest(latency)
            print(" done")

            # Test retrieval performance
            print(f"    Testing retrieval...", end="", flush=True)
            queries = QueryGenerator.generate_queries(100)

            for query in queries:
                # Pick random user/session
                user_id, session_id, _ = conversations[0]

                # Test all token budgets
                for budget in self.config.token_budgets:
                    result = adapter.retrieve_context(
                        user_id, session_id, query, budget
                    )
                    metrics.add_retrieval(
                        result.retrieval_latency_ms,
                        result.token_count
                    )
            print(" done")

            results[system_name] = {
                "ingest_avg_ms": sum(metrics.ingest_latencies_ms) / len(metrics.ingest_latencies_ms),
                "retrieval_p50_ms": metrics.get_p50_latency(),
                "retrieval_p95_ms": metrics.get_p95_latency(),
                "retrieval_p99_ms": metrics.get_p99_latency(),
                "avg_tokens": metrics.get_avg_tokens(),
                "p95_tokens": metrics.get_p95_tokens(),
            }

            print(f"    Results:")
            print(f"      Ingest (avg):     {results[system_name]['ingest_avg_ms']:.2f}ms")
            print(f"      Retrieval p50:    {results[system_name]['retrieval_p50_ms']:.2f}ms")
            print(f"      Retrieval p95:    {results[system_name]['retrieval_p95_ms']:.2f}ms")
            print(f"      Retrieval p99:    {results[system_name]['retrieval_p99_ms']:.2f}ms")
            print(f"      Context (avg):    {results[system_name]['avg_tokens']:.0f} tokens")

        return results

    def phase2_token_efficiency(self):
        """
        Phase 2: Token Efficiency

        Test context assembly within token budgets
        """
        print("\n" + "="*70)
        print("  PHASE 2: TOKEN EFFICIENCY")
        print("="*70)

        results = {}

        for system_name, adapter in self.adapters.items():
            print(f"\n  [{system_name}]")

            # Reset and load data
            adapter.reset()

            print(f"    Loading test data...", end="", flush=True)
            conversations = SyntheticWorkloadGenerator.generate_users_and_sessions(
                num_users=50,
                sessions_per_user=10,
                messages_per_session=100
            )

            for user_id, session_id, messages in conversations[:10]:  # Load subset
                adapter.ingest_messages(user_id, session_id, messages)
            print(" done")

            # Test different token budgets
            budget_results = {}

            for budget in self.config.token_budgets:
                print(f"    Testing budget={budget}...", end="", flush=True)
                metrics = BenchmarkMetrics()

                queries = QueryGenerator.generate_queries(50)

                for query in queries:
                    user_id, session_id, _ = conversations[0]

                    result = adapter.retrieve_context(
                        user_id, session_id, query, budget
                    )

                    metrics.add_retrieval(
                        result.retrieval_latency_ms,
                        result.token_count
                    )

                    # Track budget usage
                    if result.token_count > budget:
                        metrics.truncation_events += 1

                budget_results[budget] = {
                    "avg_tokens": metrics.get_avg_tokens(),
                    "p95_tokens": metrics.get_p95_tokens(),
                    "truncation_rate": metrics.truncation_events / len(queries),
                    "avg_latency_ms": sum(metrics.retrieval_latencies_ms) / len(metrics.retrieval_latencies_ms),
                }

                print(f" {budget_results[budget]['avg_tokens']:.0f} tokens avg")

            results[system_name] = budget_results

        return results

    def phase3_locomo_quality(self):
        """
        Phase 3: LoCoMo Quality Benchmark

        Test end-to-end QA accuracy
        """
        print("\n" + "="*70)
        print("  PHASE 3: LoCoMo QUALITY BENCHMARK")
        print("="*70)
        print("  (Using synthetic LoCoMo-style dataset)")

        results = {}

        # Load LoCoMo dataset
        print(f"\n  Loading dataset...", end="", flush=True)
        dataset = LoCoMoDatasetLoader.load_dataset()
        print(f" {len(dataset)} conversations")

        for system_name, adapter in self.adapters.items():
            print(f"\n  [{system_name}]")
            metrics = BenchmarkMetrics()

            # Reset and ingest conversations
            adapter.reset()

            print(f"    Ingesting conversations...", end="", flush=True)
            for conv in dataset[:20]:  # Subset for speed
                user_id = f"locomo_user_{conv['conversation_id']}"
                session_id = conv['conversation_id']

                adapter.ingest_messages(user_id, session_id, conv['messages'])
            print(" done")

            # Run QA tasks
            print(f"    Running QA tasks...", end="", flush=True)
            correct = 0
            total = 0

            for conv in dataset[:20]:
                user_id = f"locomo_user_{conv['conversation_id']}"
                session_id = conv['conversation_id']

                for qa in conv['qa_pairs']:
                    # Retrieve context
                    result = adapter.retrieve_context(
                        user_id, session_id,
                        qa['question'],
                        token_budget=2000
                    )

                    # Simple scoring: check if answer appears in context
                    if qa['answer'].lower() in result.context_str.lower():
                        correct += 1
                    total += 1

            accuracy = correct / total if total > 0 else 0.0
            print(f" {accuracy:.1%} accuracy")

            results[system_name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }

        return results

    def phase4_scale_test(self):
        """
        Phase 4: Scale Test

        Test performance under production-like load
        """
        print("\n" + "="*70)
        print("  PHASE 4: SCALE TEST")
        print("="*70)

        results = {}

        scales = [100, 500, 1000, 2000]

        for system_name, adapter in self.adapters.items():
            print(f"\n  [{system_name}]")
            scale_results = {}

            for scale in scales:
                print(f"    Scale: {scale} observations...", end="", flush=True)

                adapter.reset()

                # Generate and ingest data
                conversations = SyntheticWorkloadGenerator.generate_users_and_sessions(
                    num_users=10,
                    sessions_per_user=scale // 10,
                    messages_per_session=10
                )

                for user_id, session_id, messages in conversations:
                    adapter.ingest_messages(user_id, session_id, messages)

                # Test retrieval performance
                metrics = BenchmarkMetrics()
                queries = QueryGenerator.generate_queries(50)

                for query in queries:
                    user_id, session_id, _ = conversations[0]
                    result = adapter.retrieve_context(
                        user_id, session_id, query, 2000
                    )
                    metrics.add_retrieval(
                        result.retrieval_latency_ms,
                        result.token_count
                    )

                scale_results[scale] = {
                    "p50_ms": metrics.get_p50_latency(),
                    "p95_ms": metrics.get_p95_latency(),
                    "p99_ms": metrics.get_p99_latency(),
                }

                print(f" p95={scale_results[scale]['p95_ms']:.2f}ms")

            results[system_name] = scale_results

        return results

    def run_all_phases(self):
        """Run all benchmark phases"""
        print("="*70)
        print("  MEMORY SYSTEM COMPARISON BENCHMARK")
        print("  ToonDB vs Zep")
        print("="*70)
        print(f"\n  Systems under test: {', '.join(self.adapters.keys())}")
        print(f"  Configuration:")
        print(f"    Users: {self.config.num_users}")
        print(f"    Sessions per user: {self.config.sessions_per_user}")
        print(f"    Token budgets: {self.config.token_budgets}")

        all_results = {}

        # Phase 1: Microbenchmarks
        all_results["phase1_microbenchmarks"] = self.phase1_microbenchmarks()

        # Phase 2: Token Efficiency
        all_results["phase2_token_efficiency"] = self.phase2_token_efficiency()

        # Phase 3: LoCoMo Quality
        all_results["phase3_locomo_quality"] = self.phase3_locomo_quality()

        # Phase 4: Scale Test
        all_results["phase4_scale_test"] = self.phase4_scale_test()

        # Save results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.config.results_dir}/comparison_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to: {output_file}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print final summary"""
        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)

        # Microbenchmarks
        print("\n  Phase 1: Microbenchmarks (Retrieval Latency)")
        print(f"  {'System':<12} {'p50 (ms)':<10} {'p95 (ms)':<10} {'p99 (ms)':<10}")
        print("  " + "-"*42)

        for system, data in results["phase1_microbenchmarks"].items():
            print(f"  {system:<12} {data['retrieval_p50_ms']:<10.2f} "
                  f"{data['retrieval_p95_ms']:<10.2f} {data['retrieval_p99_ms']:<10.2f}")

        # Token Efficiency
        print("\n  Phase 2: Token Efficiency (2k budget)")
        print(f"  {'System':<12} {'Avg Tokens':<12} {'Truncation':<12}")
        print("  " + "-"*36)

        for system, budgets in results["phase2_token_efficiency"].items():
            if 2000 in budgets:
                data = budgets[2000]
                print(f"  {system:<12} {data['avg_tokens']:<12.0f} "
                      f"{data['truncation_rate']:<12.1%}")

        # LoCoMo Quality
        print("\n  Phase 3: LoCoMo Quality (QA Accuracy)")
        print(f"  {'System':<12} {'Accuracy':<12}")
        print("  " + "-"*24)

        for system, data in results["phase3_locomo_quality"].items():
            print(f"  {system:<12} {data['accuracy']:<12.1%}")

        # Scale Test
        print("\n  Phase 4: Scale Test (p95 latency at 1000 obs)")
        print(f"  {'System':<12} {'p95 (ms)':<12}")
        print("  " + "-"*24)

        for system, scales in results["phase4_scale_test"].items():
            if 1000 in scales:
                print(f"  {system:<12} {scales[1000]['p95_ms']:<12.2f}")

        print("\n" + "="*70)

    def cleanup(self):
        """Cleanup adapters"""
        for adapter in self.adapters.values():
            adapter.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run memory system comparison"""
    # Check prerequisites
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY not set")
        sys.exit(1)

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Error: AZURE_OPENAI_ENDPOINT not set")
        sys.exit(1)

    # Get config
    config = get_benchmark_config()

    # Run benchmark
    runner = MemoryBenchmarkRunner(config)

    try:
        runner.run_all_phases()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
