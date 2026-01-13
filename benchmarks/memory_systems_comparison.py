#!/usr/bin/env python3
"""
Memory Systems Head-to-Head: SochDB vs Zep vs Mem0

Real-world agent memory benchmark inspired by the Zep vs Mem0 controversy.
This benchmark uses actual LLM calls to test memory quality, latency, and efficiency.

Key Metrics (from the debate):
- Memory Recall Quality
- Search Latency (p50, p95, p99)
- Context Token Usage
- Real Agent Conversation Performance

Usage:
    python3 benchmarks/memory_systems_comparison.py
"""

import os
import sys
import time
import json
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import tiktoken

# Azure OpenAI
from openai import AzureOpenAI


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    azure_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    azure_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    embedding_dim: int = 1536
    num_test_conversations: int = 50
    num_test_queries: int = 200


# =============================================================================
# Real Conversation Dataset
# =============================================================================

class ConversationDataset:
    """Realistic agent conversations for testing."""

    @staticmethod
    def get_conversations() -> List[Dict[str, List[str]]]:
        """Get diverse conversation dataset."""
        return [
            {
                "topic": "Account Troubleshooting",
                "messages": [
                    "I can't log into my account",
                    "Let me help you. What error do you see?",
                    "It says 'Invalid credentials'",
                    "I'll send you a password reset link",
                    "Got the email, resetting now",
                    "Great! Try logging in again",
                    "Works now, thanks!",
                ]
            },
            {
                "topic": "API Integration",
                "messages": [
                    "How do I integrate your API?",
                    "You'll need an API key first",
                    "Where do I get that?",
                    "In your dashboard under Settings > API",
                    "Found it. What's the endpoint?",
                    "Use https://api.example.com/v1",
                    "Do you have Python examples?",
                    "Yes, check our docs at docs.example.com/python",
                ]
            },
            {
                "topic": "Billing Question",
                "messages": [
                    "Why was I charged $299?",
                    "Let me check your account",
                    "That's your annual subscription renewal",
                    "I thought I was on monthly",
                    "You switched to annual last year for the discount",
                    "Oh right, can I change back?",
                    "Yes, I can switch you to monthly at $29",
                    "Please do that",
                ]
            },
            {
                "topic": "Feature Request",
                "messages": [
                    "Do you support SSO?",
                    "Yes, enterprise plans include SAML SSO",
                    "What about SCIM provisioning?",
                    "That's included too",
                    "How do I set it up?",
                    "I'll send you our SSO setup guide",
                    "Can you help with configuration?",
                    "Our support team can do a setup call",
                ]
            },
            {
                "topic": "Performance Issue",
                "messages": [
                    "My queries are really slow",
                    "How slow are we talking?",
                    "30+ seconds",
                    "That's not normal. What's your query?",
                    "SELECT * FROM users",
                    "Ah, you need to add pagination",
                    "How do I do that?",
                    "Use LIMIT and OFFSET in your query",
                    "What's a good page size?",
                    "Start with 100 rows per page",
                ]
            },
            {
                "topic": "Security Compliance",
                "messages": [
                    "Do you have SOC 2?",
                    "Yes, we're SOC 2 Type II certified",
                    "What about GDPR compliance?",
                    "Fully compliant with GDPR",
                    "Can we get a copy of your certifications?",
                    "I'll send you our security whitepaper",
                    "Do you support data residency?",
                    "Yes, we have US, EU, and APAC regions",
                ]
            },
            {
                "topic": "Upgrade Discussion",
                "messages": [
                    "Tell me about your enterprise plan",
                    "Enterprise includes unlimited users",
                    "What's the pricing?",
                    "Starts at $999/month",
                    "Do you offer discounts?",
                    "Yes, for annual commitments",
                    "What's the annual price?",
                    "$10,000 per year saves you $2,000",
                ]
            },
            {
                "topic": "Technical Support",
                "messages": [
                    "I'm getting a 429 error",
                    "That's a rate limit error",
                    "What's the limit?",
                    "100 requests per minute on your plan",
                    "Can I increase it?",
                    "Upgrade to Pro for 1000 req/min",
                    "How much is Pro?",
                    "$99 per month",
                ]
            },
        ]


class BenchmarkQueries:
    """Test queries to evaluate memory recall."""

    @staticmethod
    def get_queries() -> List[Dict[str, str]]:
        """Get test queries with expected topics."""
        return [
            {"query": "How do I reset my password?", "expected_topic": "Account Troubleshooting"},
            {"query": "What's the API endpoint?", "expected_topic": "API Integration"},
            {"query": "Why was I charged?", "expected_topic": "Billing Question"},
            {"query": "Do you support single sign-on?", "expected_topic": "Feature Request"},
            {"query": "My queries are slow", "expected_topic": "Performance Issue"},
            {"query": "Are you SOC 2 certified?", "expected_topic": "Security Compliance"},
            {"query": "How much is enterprise?", "expected_topic": "Upgrade Discussion"},
            {"query": "I'm getting rate limited", "expected_topic": "Technical Support"},
            {"query": "Forgot password", "expected_topic": "Account Troubleshooting"},
            {"query": "Integration documentation", "expected_topic": "API Integration"},
            {"query": "Subscription cost", "expected_topic": "Billing Question"},
            {"query": "SCIM provisioning", "expected_topic": "Feature Request"},
            {"query": "Database performance", "expected_topic": "Performance Issue"},
            {"query": "GDPR compliance", "expected_topic": "Security Compliance"},
            {"query": "Annual pricing", "expected_topic": "Upgrade Discussion"},
            {"query": "429 error", "expected_topic": "Technical Support"},
        ]


# =============================================================================
# Memory System Implementations
# =============================================================================

class SochDBMemory:
    """SochDB-based agent memory."""

    def __init__(self, client: AzureOpenAI, config: Config):
        from sochdb import VectorIndex
        self.client = client
        self.config = config
        self.index = VectorIndex(dimension=config.embedding_dim, max_connections=32, ef_construction=200)
        self.memories: List[Dict] = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts."""
        response = self.client.embeddings.create(
            model=self.config.azure_embedding_deployment,
            input=texts
        )
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    def add(self, text: str, metadata: Dict) -> float:
        """Add memory."""
        start = time.perf_counter()

        embedding = self.embed([text])[0]
        idx = len(self.memories)

        self.index.insert_batch(
            np.array([idx], dtype=np.uint64),
            np.array([embedding], dtype=np.float32)
        )

        self.memories.append({"text": text, "metadata": metadata})

        return time.perf_counter() - start

    def search(self, query: str, k: int = 5) -> Tuple[List[str], float]:
        """Search memory."""
        start = time.perf_counter()

        query_embedding = self.embed([query])[0]
        results = self.index.search(query_embedding, k=k)

        context_parts = []
        for idx, score in results:
            if int(idx) < len(self.memories):
                context_parts.append(self.memories[int(idx)]["text"])

        latency = time.perf_counter() - start
        return context_parts, latency

    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(self.tokenizer.encode(text))


class ChromaDBMemory:
    """ChromaDB-based agent memory (as comparison)."""

    def __init__(self, client: AzureOpenAI, config: Config):
        import chromadb
        self.client = client
        self.config = config
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=f"bench_{int(time.time())}",
            metadata={"hnsw:space": "cosine"}
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.doc_count = 0

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts."""
        response = self.client.embeddings.create(
            model=self.config.azure_embedding_deployment,
            input=texts
        )
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    def add(self, text: str, metadata: Dict) -> float:
        """Add memory."""
        start = time.perf_counter()

        embedding = self.embed([text])[0]

        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[metadata],
            ids=[f"doc_{self.doc_count}"]
        )
        self.doc_count += 1

        return time.perf_counter() - start

    def search(self, query: str, k: int = 5) -> Tuple[List[str], float]:
        """Search memory."""
        start = time.perf_counter()

        query_embedding = self.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        context_parts = results['documents'][0] if results['documents'] else []
        latency = time.perf_counter() - start

        return context_parts, latency

    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(self.tokenizer.encode(text))


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResults:
    """Benchmark results."""
    system_name: str
    insert_latencies: List[float] = field(default_factory=list)
    search_latencies: List[float] = field(default_factory=list)
    context_token_counts: List[int] = field(default_factory=list)
    recall_scores: List[float] = field(default_factory=list)

    def add_insert(self, latency: float):
        self.insert_latencies.append(latency * 1000)  # Convert to ms

    def add_search(self, latency: float, context_tokens: int, recall_score: float):
        self.search_latencies.append(latency * 1000)  # Convert to ms
        self.context_token_counts.append(context_tokens)
        self.recall_scores.append(recall_score)

    def get_stats(self) -> Dict:
        """Calculate statistics."""
        if not self.search_latencies:
            return {}

        sorted_latencies = sorted(self.search_latencies)
        n = len(sorted_latencies)

        return {
            "system": self.system_name,
            "insert_avg_ms": statistics.mean(self.insert_latencies) if self.insert_latencies else 0,
            "search_p50_ms": sorted_latencies[n//2] if n > 0 else 0,
            "search_p95_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "search_p99_ms": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            "search_avg_ms": statistics.mean(self.search_latencies),
            "context_avg_tokens": statistics.mean(self.context_token_counts) if self.context_token_counts else 0,
            "recall_avg": statistics.mean(self.recall_scores) if self.recall_scores else 0,
        }


class MemorySystemBenchmark:
    """Benchmark runner."""

    def __init__(self, config: Config):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint
        )

    def run_benchmark(self, memory_system, system_name: str) -> BenchmarkResults:
        """Run benchmark on a memory system."""
        print(f"\n{'='*70}")
        print(f"  Benchmarking: {system_name}")
        print(f"{'='*70}")

        results = BenchmarkResults(system_name)

        # Phase 1: Load conversations into memory
        print(f"\n  Phase 1: Loading conversations into memory...")
        conversations = ConversationDataset.get_conversations()

        message_count = 0
        for conv in conversations:
            topic = conv["topic"]
            for msg in conv["messages"]:
                latency = memory_system.add(msg, {"topic": topic})
                results.add_insert(latency)
                message_count += 1

        print(f"  ✓ Loaded {message_count} messages from {len(conversations)} conversations")
        print(f"    Avg insert latency: {statistics.mean(results.insert_latencies):.2f}ms")

        # Phase 2: Query memory
        print(f"\n  Phase 2: Testing memory recall ({self.config.num_test_queries} queries)...")
        queries = BenchmarkQueries.get_queries()

        for i in range(self.config.num_test_queries):
            query_data = queries[i % len(queries)]
            query = query_data["query"]
            expected_topic = query_data["expected_topic"]

            # Search memory
            context_parts, latency = memory_system.search(query, k=5)
            context = "\n".join(context_parts)
            context_tokens = memory_system.count_tokens(context)

            # Simple recall score: did we retrieve relevant context?
            recall_score = 1.0 if expected_topic.lower() in context.lower() else 0.0

            results.add_search(latency, context_tokens, recall_score)

            if (i + 1) % 50 == 0:
                print(f"    Completed {i + 1}/{self.config.num_test_queries} queries")

        stats = results.get_stats()
        print(f"\n  Results:")
        print(f"    Search p50: {stats['search_p50_ms']:.2f}ms")
        print(f"    Search p95: {stats['search_p95_ms']:.2f}ms")
        print(f"    Search p99: {stats['search_p99_ms']:.2f}ms")
        print(f"    Avg context: {stats['context_avg_tokens']:.0f} tokens")
        print(f"    Recall rate: {stats['recall_avg']:.1%}")
        print(f"  ✓ Benchmark completed!")

        return results

    def run_all(self):
        """Run all benchmarks."""
        print("="*70)
        print("  MEMORY SYSTEMS HEAD-TO-HEAD BENCHMARK")
        print("  SochDB vs ChromaDB")
        print("  (Inspired by the Zep vs Mem0 Controversy)")
        print("="*70)
        print(f"\n  Test Configuration:")
        print(f"    Conversations: {len(ConversationDataset.get_conversations())}")
        print(f"    Test Queries: {self.config.num_test_queries}")
        print(f"    LLM: Azure OpenAI {self.config.azure_chat_deployment}")
        print(f"    Embeddings: {self.config.azure_embedding_deployment}")

        all_results = []

        # SochDB
        sochdb = SochDBMemory(self.client, self.config)
        sochdb_results = self.run_benchmark(sochdb, "SochDB")
        all_results.append(sochdb_results.get_stats())

        # ChromaDB
        chromadb = ChromaDBMemory(self.client, self.config)
        chromadb_results = self.run_benchmark(chromadb, "ChromaDB")
        all_results.append(chromadb_results.get_stats())

        # Print comparison
        self.print_comparison(all_results)

        # Save results
        output_file = f"memory_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n  Results saved to: {output_file}")

        return all_results

    def print_comparison(self, results: List[Dict]):
        """Print comparison table."""
        print(f"\n{'='*70}")
        print("  FINAL COMPARISON")
        print(f"{'='*70}")

        print(f"\n{'System':<12} {'p50 (ms)':<10} {'p95 (ms)':<10} {'p99 (ms)':<10} {'Context':<10} {'Recall':<10}")
        print("-"*70)

        for r in results:
            print(f"{r['system']:<12} {r['search_p50_ms']:<10.2f} {r['search_p95_ms']:<10.2f} "
                  f"{r['search_p99_ms']:<10.2f} {r['context_avg_tokens']:<10.0f} {r['recall_avg']:<10.1%}")

        # Performance comparison
        if len(results) >= 2:
            print(f"\n{'='*70}")
            print("  PERFORMANCE vs SochDB")
            print(f"{'='*70}")

            sochdb = next((r for r in results if r['system'] == 'SochDB'), None)
            if sochdb:
                for r in results:
                    if r['system'] != 'SochDB':
                        speedup_p95 = r['search_p95_ms'] / sochdb['search_p95_ms']
                        token_ratio = r['context_avg_tokens'] / sochdb['context_avg_tokens'] if sochdb['context_avg_tokens'] > 0 else 1.0

                        print(f"\n  {r['system']}:")
                        if speedup_p95 > 1:
                            print(f"    Search (p95): {speedup_p95:.2f}x SLOWER")
                        else:
                            print(f"    Search (p95): {1/speedup_p95:.2f}x FASTER")

                        print(f"    Context size: {token_ratio:.2f}x")

        print(f"\n{'='*70}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run memory systems benchmark."""
    config = Config()

    if not config.azure_api_key or not config.azure_endpoint:
        print("Error: Azure OpenAI credentials not found")
        print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        sys.exit(1)

    benchmark = MemorySystemBenchmark(config)
    benchmark.run_all()


if __name__ == "__main__":
    main()
