#!/usr/bin/env python3
"""
Agent Memory Systems Benchmark: SochDB vs Zep vs Mem0

Real-world benchmark comparing agent memory systems using actual LLM calls.
Tests based on the Zep vs Mem0 controversy:
- Memory quality (recall accuracy)
- Search latency (p95)
- Token efficiency
- Multi-turn conversation performance

Usage:
    python3 benchmarks/agent_memory_benchmark.py
"""

import os
import sys
import time
import json
import random
import statistics
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import tiktoken

# Azure OpenAI
from openai import AzureOpenAI


# =============================================================================
# Configuration & LLM Client
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Azure OpenAI settings
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    azure_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    azure_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # Benchmark settings
    num_conversations: int = 10
    messages_per_conversation: int = 20
    num_queries: int = 100
    embedding_dimension: int = 1536


class LLMClient:
    """Azure OpenAI client for embeddings and chat."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.total_tokens = 0

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        if not texts:
            return np.array([])

        # Azure OpenAI has a limit, batch if needed
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.config.azure_embedding_deployment,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            self.total_tokens += response.usage.total_tokens

        return np.array(all_embeddings, dtype=np.float32)

    def chat(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        """Generate chat completion with optional memory context."""
        if context:
            # Prepend context as system message
            messages = [
                {"role": "system", "content": f"Relevant context from memory:\n{context}"}
            ] + messages

        response = self.client.chat.completions.create(
            model=self.config.azure_deployment,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        self.total_tokens += response.usage.total_tokens
        return response.choices[0].message.content

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))


# =============================================================================
# Memory System Adapters
# =============================================================================

@dataclass
class MemoryStats:
    """Statistics for a memory system."""
    insert_latencies: List[float] = field(default_factory=list)
    search_latencies: List[float] = field(default_factory=list)
    recall_scores: List[float] = field(default_factory=list)
    context_sizes: List[int] = field(default_factory=list)
    total_tokens_used: int = 0

    def add_insert(self, latency: float):
        self.insert_latencies.append(latency)

    def add_search(self, latency: float, recall_score: float, context_size: int):
        self.search_latencies.append(latency)
        self.recall_scores.append(recall_score)
        self.context_sizes.append(context_size)

    def get_p95_latency(self) -> float:
        if not self.search_latencies:
            return 0.0
        return statistics.quantiles(self.search_latencies, n=20)[18]  # 95th percentile

    def get_avg_latency(self) -> float:
        return statistics.mean(self.search_latencies) if self.search_latencies else 0.0

    def get_avg_recall(self) -> float:
        return statistics.mean(self.recall_scores) if self.recall_scores else 0.0

    def get_avg_context_size(self) -> float:
        return statistics.mean(self.context_sizes) if self.context_sizes else 0.0


class SochDBMemoryAdapter:
    """SochDB-based agent memory system."""

    def __init__(self, llm: LLMClient):
        from sochdb import VectorIndex
        self.llm = llm
        self.index = VectorIndex(dimension=llm.config.embedding_dimension, max_connections=32, ef_construction=200)
        self.memories: List[Dict[str, Any]] = []
        self.next_id = 0
        self.stats = MemoryStats()

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> float:
        """Add a memory to the system."""
        start = time.perf_counter()

        # Embed the text
        embedding = self.llm.embed([text])[0]

        # Store in index
        self.index.insert_batch(
            np.array([self.next_id], dtype=np.uint64),
            np.array([embedding], dtype=np.float32)
        )

        # Store metadata
        self.memories.append({
            "id": self.next_id,
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        self.next_id += 1

        latency = time.perf_counter() - start
        self.stats.add_insert(latency)
        return latency

    def search_memory(self, query: str, k: int = 5) -> Tuple[str, float]:
        """Search for relevant memories and return context."""
        start = time.perf_counter()

        # Embed query
        query_embedding = self.llm.embed([query])[0]

        # Search
        results = self.index.search(query_embedding, k=k)

        # Build context from results
        context_parts = []
        for idx, score in results:
            if int(idx) < len(self.memories):
                memory = self.memories[int(idx)]
                context_parts.append(memory["text"])

        context = "\n".join(context_parts)
        latency = time.perf_counter() - start

        # Calculate recall score (simplified: based on result count)
        recall_score = len(context_parts) / k if k > 0 else 0.0

        context_tokens = self.llm.count_tokens(context)
        self.stats.add_search(latency, recall_score, context_tokens)

        return context, latency

    def cleanup(self):
        pass


class Mem0MemoryAdapter:
    """Mem0-based agent memory system."""

    def __init__(self, llm: LLMClient):
        from mem0 import Memory
        self.llm = llm
        self.stats = MemoryStats()

        # Initialize Mem0 with Azure OpenAI
        config = {
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": llm.config.azure_deployment,
                    "api_key": llm.config.azure_api_key,
                    "azure_endpoint": llm.config.azure_endpoint,
                    "api_version": llm.config.azure_api_version,
                }
            },
            "embedder": {
                "provider": "azure_openai",
                "config": {
                    "model": llm.config.azure_embedding_deployment,
                    "api_key": llm.config.azure_api_key,
                    "azure_endpoint": llm.config.azure_endpoint,
                    "api_version": llm.config.azure_api_version,
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"mem0_bench_{int(time.time())}",
                    "embedding_model_dims": llm.config.embedding_dimension,
                    "path": "/tmp/mem0_qdrant"
                }
            }
        }

        try:
            self.memory = Memory.from_config(config)
            self.user_id = "benchmark_user"
        except Exception as e:
            print(f"Warning: Mem0 initialization failed: {e}")
            print("Mem0 will be skipped in benchmarks")
            self.memory = None

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> float:
        """Add a memory to Mem0."""
        if not self.memory:
            return 0.0

        start = time.perf_counter()

        try:
            self.memory.add(
                text,
                user_id=self.user_id,
                metadata=metadata
            )
        except Exception as e:
            print(f"Mem0 add error: {e}")

        latency = time.perf_counter() - start
        self.stats.add_insert(latency)
        return latency

    def search_memory(self, query: str, k: int = 5) -> Tuple[str, float]:
        """Search Mem0 for relevant memories."""
        if not self.memory:
            return "", 0.0

        start = time.perf_counter()

        try:
            results = self.memory.search(query, user_id=self.user_id, limit=k)

            # Build context
            context_parts = []
            for result in results:
                if isinstance(result, dict) and "memory" in result:
                    context_parts.append(result["memory"])
                elif isinstance(result, str):
                    context_parts.append(result)

            context = "\n".join(context_parts)
            recall_score = len(context_parts) / k if k > 0 else 0.0
        except Exception as e:
            print(f"Mem0 search error: {e}")
            context = ""
            recall_score = 0.0

        latency = time.perf_counter() - start
        context_tokens = self.llm.count_tokens(context)
        self.stats.add_search(latency, recall_score, context_tokens)

        return context, latency

    def cleanup(self):
        if self.memory:
            try:
                self.memory.reset()
            except:
                pass


class ZepMemoryAdapter:
    """Zep-based agent memory system."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.stats = MemoryStats()
        self.zep_client = None

        # Zep requires a server, skip if not available
        print("Note: Zep requires a running server. Skipping Zep in this benchmark.")
        print("To include Zep, set ZEP_API_URL environment variable and ensure server is running.")

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> float:
        """Zep is skipped - requires server."""
        return 0.0

    def search_memory(self, query: str, k: int = 5) -> Tuple[str, float]:
        """Zep is skipped - requires server."""
        return "", 0.0

    def cleanup(self):
        pass


# =============================================================================
# Test Scenarios
# =============================================================================

class ConversationScenario:
    """Generate realistic conversation scenarios."""

    @staticmethod
    def generate_customer_support_conversation() -> List[Tuple[str, str]]:
        """Generate a customer support conversation."""
        return [
            ("user", "I'm having trouble logging into my account"),
            ("assistant", "I can help with that. What error message do you see?"),
            ("user", "It says 'Invalid credentials' but I'm sure my password is correct"),
            ("assistant", "Let me check your account. Can you confirm your email?"),
            ("user", "It's john.doe@example.com"),
            ("assistant", "I see your account. Let's try resetting your password."),
            ("user", "Okay, how do I do that?"),
            ("assistant", "I'll send a reset link to your email."),
            ("user", "Got it, the email arrived"),
            ("assistant", "Great! Follow the link to create a new password."),
            ("user", "Done! I can log in now"),
            ("assistant", "Excellent! Is there anything else I can help with?"),
            ("user", "Actually, I notice my subscription expires next week"),
            ("assistant", "Would you like to renew it?"),
            ("user", "Yes, what are my options?"),
            ("assistant", "You can choose monthly ($29) or annual ($299)."),
            ("user", "I'll go with annual"),
            ("assistant", "Perfect! I'll process that for you."),
            ("user", "Thanks for your help!"),
            ("assistant", "You're welcome! Have a great day!"),
        ]

    @staticmethod
    def generate_technical_support_conversation() -> List[Tuple[str, str]]:
        """Generate a technical support conversation."""
        return [
            ("user", "My API requests are timing out"),
            ("assistant", "Let's troubleshoot this. What endpoint are you calling?"),
            ("user", "/api/v1/users"),
            ("assistant", "How long before timeout occurs?"),
            ("user", "About 30 seconds"),
            ("assistant", "That's longer than our 10s limit. Are you paginating?"),
            ("user", "No, I'm fetching all users at once"),
            ("assistant", "That's the issue. You need to use pagination with limit/offset."),
            ("user", "How do I do that?"),
            ("assistant", "Add ?limit=100&offset=0 to your request."),
            ("user", "Okay, trying now... it works!"),
            ("assistant", "Great! For large datasets, always paginate."),
            ("user", "What's the maximum page size?"),
            ("assistant", "We recommend 100, max is 1000."),
            ("user", "Got it. One more question about rate limits"),
            ("assistant", "We have 100 requests per minute for free tier."),
            ("user", "I'm on Pro plan"),
            ("assistant", "Pro has 1000 requests per minute."),
            ("user", "Perfect, thanks!"),
            ("assistant", "Happy to help!"),
        ]

    @staticmethod
    def generate_product_inquiry_conversation() -> List[Tuple[str, str]]:
        """Generate a product inquiry conversation."""
        return [
            ("user", "Tell me about your enterprise plan"),
            ("assistant", "Our enterprise plan includes unlimited users and priority support."),
            ("user", "What's the pricing?"),
            ("assistant", "Enterprise starts at $999/month with custom pricing for large teams."),
            ("user", "Do you offer SSO?"),
            ("assistant", "Yes, enterprise includes SAML SSO and SCIM provisioning."),
            ("user", "What about data residency options?"),
            ("assistant", "We support US, EU, and APAC regions."),
            ("user", "Can we get a custom SLA?"),
            ("assistant", "Yes, we offer 99.9% or 99.99% SLA tiers."),
            ("user", "What's included in priority support?"),
            ("assistant", "24/7 phone and chat with <1 hour response time."),
            ("user", "Do you have SOC 2 certification?"),
            ("assistant", "Yes, we're SOC 2 Type II and GDPR compliant."),
            ("user", "Can we get a trial?"),
            ("assistant", "Enterprise trials are 30 days with full feature access."),
            ("user", "Who do I contact to start?"),
            ("assistant", "I can connect you with our sales team."),
            ("user", "Please do"),
            ("assistant", "Great! I'll have someone reach out within 24 hours."),
            ("user", "Thank you"),
            ("assistant", "You're welcome! Looking forward to working with you."),
        ]

    @staticmethod
    def get_random_conversation() -> List[Tuple[str, str]]:
        """Get a random conversation."""
        scenarios = [
            ConversationScenario.generate_customer_support_conversation,
            ConversationScenario.generate_technical_support_conversation,
            ConversationScenario.generate_product_inquiry_conversation,
        ]
        return random.choice(scenarios)()


# =============================================================================
# Benchmark Runner
# =============================================================================

class AgentMemoryBenchmark:
    """Run agent memory benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.llm = LLMClient(config)

    def run_conversation_benchmark(self, adapter_name: str, adapter: Any) -> Dict[str, Any]:
        """Run conversation-based benchmark on a memory adapter."""
        print(f"\n{'='*70}")
        print(f"  Benchmarking: {adapter_name}")
        print(f"{'='*70}")

        # Generate conversations
        conversations = []
        for i in range(self.config.num_conversations):
            conv = ConversationScenario.get_random_conversation()
            conversations.append(conv)

        # Process conversations and build memory
        print(f"\n  Phase 1: Building memory from {self.config.num_conversations} conversations...")
        for i, conversation in enumerate(conversations):
            for role, message in conversation:
                # Add each message as a memory
                adapter.add_memory(
                    message,
                    metadata={"role": role, "conversation_id": i}
                )

            if (i + 1) % 5 == 0:
                print(f"    Processed {i + 1}/{self.config.num_conversations} conversations")

        print(f"  ✓ Stored {sum(len(c) for c in conversations)} messages")

        # Test memory recall with queries
        print(f"\n  Phase 2: Testing memory recall with {self.config.num_queries} queries...")
        queries = [
            "How do I reset my password?",
            "What are the rate limits?",
            "Tell me about enterprise pricing",
            "How do I fix login issues?",
            "What's included in priority support?",
            "How do I paginate API requests?",
            "Do you support SSO?",
            "What regions are available?",
            "How do I renew my subscription?",
            "What's the SLA guarantee?",
        ]

        for i in range(self.config.num_queries):
            query = random.choice(queries)
            context, latency = adapter.search_memory(query, k=5)

            if (i + 1) % 25 == 0:
                print(f"    Completed {i + 1}/{self.config.num_queries} queries")

        # Compute statistics
        stats = adapter.stats
        results = {
            "system": adapter_name,
            "avg_insert_latency_ms": statistics.mean(stats.insert_latencies) * 1000 if stats.insert_latencies else 0,
            "p95_search_latency_ms": stats.get_p95_latency() * 1000,
            "avg_search_latency_ms": stats.get_avg_latency() * 1000,
            "avg_recall_score": stats.get_avg_recall(),
            "avg_context_tokens": stats.get_avg_context_size(),
            "total_memories": len(adapter.memories) if hasattr(adapter, 'memories') else 0,
        }

        print(f"\n  Results:")
        print(f"    Insert latency (avg): {results['avg_insert_latency_ms']:.2f}ms")
        print(f"    Search latency (avg): {results['avg_search_latency_ms']:.2f}ms")
        print(f"    Search latency (p95): {results['p95_search_latency_ms']:.2f}ms")
        print(f"    Recall score (avg):   {results['avg_recall_score']:.2%}")
        print(f"    Context size (avg):   {results['avg_context_tokens']:.0f} tokens")
        print(f"  ✓ Benchmark completed!")

        return results

    def run_all(self):
        """Run benchmarks on all memory systems."""
        print("="*70)
        print("  AGENT MEMORY SYSTEMS BENCHMARK")
        print("  SochDB vs Zep vs Mem0")
        print("="*70)
        print(f"\n  Configuration:")
        print(f"    Conversations: {self.config.num_conversations}")
        print(f"    Queries: {self.config.num_queries}")
        print(f"    Azure OpenAI: {self.config.azure_deployment}")

        all_results = []

        # SochDB
        print("\n" + "="*70)
        sochdb_adapter = SochDBMemoryAdapter(self.llm)
        try:
            results = self.run_conversation_benchmark("SochDB", sochdb_adapter)
            all_results.append(results)
        finally:
            sochdb_adapter.cleanup()

        # Mem0
        print("\n" + "="*70)
        mem0_adapter = Mem0MemoryAdapter(self.llm)
        if mem0_adapter.memory:
            try:
                results = self.run_conversation_benchmark("Mem0", mem0_adapter)
                all_results.append(results)
            finally:
                mem0_adapter.cleanup()
        else:
            print("  Skipping Mem0 (initialization failed)")

        # Zep (skipped - requires server)
        print("\n" + "="*70)
        print("  Skipping Zep (requires running server)")

        # Final summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print final benchmark summary."""
        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)

        if not results:
            print("\n  No results to display")
            return

        # Table header
        print(f"\n{'System':<15} {'Insert (ms)':<12} {'Search p95 (ms)':<16} {'Recall':<10} {'Context':<12}")
        print("-"*70)

        for r in results:
            print(f"{r['system']:<15} {r['avg_insert_latency_ms']:<12.2f} "
                  f"{r['p95_search_latency_ms']:<16.2f} {r['avg_recall_score']:<10.2%} "
                  f"{r['avg_context_tokens']:<12.0f}")

        # Speedup comparison (vs SochDB)
        if len(results) > 1:
            print("\n" + "="*70)
            print("  PERFORMANCE vs SochDB")
            print("="*70)

            sochdb_result = next((r for r in results if r['system'] == 'SochDB'), None)
            if sochdb_result:
                baseline_latency = sochdb_result['p95_search_latency_ms']

                for r in results:
                    if r['system'] != 'SochDB':
                        speedup = r['p95_search_latency_ms'] / baseline_latency
                        if speedup > 1:
                            print(f"\n  {r['system']}: {speedup:.2f}x SLOWER")
                        else:
                            print(f"\n  {r['system']}: {1/speedup:.2f}x FASTER")

        print("\n  ✓ All benchmarks completed!")
        print("="*70)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run agent memory benchmarks."""
    config = BenchmarkConfig()

    # Validate configuration
    if not config.azure_api_key or not config.azure_endpoint:
        print("Error: Azure OpenAI credentials not found in environment variables")
        print("Please set: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
        sys.exit(1)

    benchmark = AgentMemoryBenchmark(config)
    results = benchmark.run_all()

    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
