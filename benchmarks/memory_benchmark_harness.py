"""
Memory System Benchmark Harness
================================

Apples-to-apples comparison of SochDB vs Zep for agent memory systems.

Both systems implement the same contract:
- ingest_messages(user_id, session_id, messages[])
- ingest_docs(tenant_id, docs[])
- retrieve_context(user_id, session_id, query_text, token_budget) -> {context_str, metadata}
- update_memory(...) / delete_memory(...)

Evaluation:
1. LoCoMo benchmark (quality + long-term memory)
2. Synthetic production workload (scale + latency)

Metrics:
- Context payload efficiency (tokens, truncation)
- Retrieval performance (latency p50/p95/p99)
- End-to-end quality (LoCoMo QA accuracy)
- Operational correctness (durability, updates)
"""

import abc
import time
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Message:
    """A single message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A knowledge document/snippet"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResult:
    """Retrieved context from memory system"""
    context_str: str
    metadata: Dict[str, Any]
    token_count: int
    retrieval_latency_ms: float
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark"""
    # Latency metrics
    retrieval_latencies_ms: List[float] = field(default_factory=list)
    ingest_latencies_ms: List[float] = field(default_factory=list)

    # Token metrics
    context_tokens: List[int] = field(default_factory=list)
    total_prompt_tokens: List[int] = field(default_factory=list)
    truncation_events: int = 0

    # Quality metrics
    qa_scores: List[float] = field(default_factory=list)

    # Operational metrics
    failures: List[str] = field(default_factory=list)

    def add_retrieval(self, latency_ms: float, tokens: int):
        self.retrieval_latencies_ms.append(latency_ms)
        self.context_tokens.append(tokens)

    def add_ingest(self, latency_ms: float):
        self.ingest_latencies_ms.append(latency_ms)

    def add_qa_score(self, score: float):
        self.qa_scores.append(score)

    def get_p50_latency(self) -> float:
        if not self.retrieval_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.retrieval_latencies_ms)
        return sorted_lat[len(sorted_lat) // 2]

    def get_p95_latency(self) -> float:
        if not self.retrieval_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.retrieval_latencies_ms)
        return sorted_lat[int(len(sorted_lat) * 0.95)]

    def get_p99_latency(self) -> float:
        if not self.retrieval_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.retrieval_latencies_ms)
        return sorted_lat[int(len(sorted_lat) * 0.99)]

    def get_avg_tokens(self) -> float:
        if not self.context_tokens:
            return 0.0
        return sum(self.context_tokens) / len(self.context_tokens)

    def get_p95_tokens(self) -> int:
        if not self.context_tokens:
            return 0
        sorted_tokens = sorted(self.context_tokens)
        return sorted_tokens[int(len(sorted_tokens) * 0.95)]

    def get_avg_qa_score(self) -> float:
        if not self.qa_scores:
            return 0.0
        return sum(self.qa_scores) / len(self.qa_scores)


# =============================================================================
# Memory System Interface (Abstract Base)
# =============================================================================

class MemorySystemAdapter(abc.ABC):
    """
    Unified interface for memory systems.

    Both SochDB and Zep must implement this contract for fair comparison.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @abc.abstractmethod
    def ingest_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[Message]
    ) -> float:
        """
        Ingest conversation messages into memory.

        Returns:
            ingest_latency_ms
        """
        pass

    @abc.abstractmethod
    def ingest_docs(
        self,
        tenant_id: str,
        docs: List[Document]
    ) -> float:
        """
        Ingest knowledge documents.

        Returns:
            ingest_latency_ms
        """
        pass

    @abc.abstractmethod
    def retrieve_context(
        self,
        user_id: str,
        session_id: str,
        query_text: str,
        token_budget: int
    ) -> ContextResult:
        """
        Retrieve context for next LLM call within token budget.

        This is the core operation being benchmarked.

        Returns:
            ContextResult with context_str, metadata, token_count, latency
        """
        pass

    @abc.abstractmethod
    def update_memory(
        self,
        user_id: str,
        session_id: str,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an existing memory.

        Returns:
            success
        """
        pass

    @abc.abstractmethod
    def delete_memory(
        self,
        user_id: str,
        session_id: str,
        memory_id: str
    ) -> bool:
        """
        Delete a memory.

        Returns:
            success
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset/clear all data (for testing)"""
        pass

    @abc.abstractmethod
    def close(self):
        """Cleanup resources"""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    # System configs
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0

    # Workload params
    num_users: int = 100
    sessions_per_user: int = 10
    messages_per_session: int = 200
    num_docs: int = 10000

    # Retrieval params
    token_budgets: List[int] = field(default_factory=lambda: [2000, 4000, 8000])
    top_k: int = 10

    # Test params
    num_queries: int = 1000
    concurrency: int = 10

    # Output
    results_dir: str = "benchmark_results"
    verbose: bool = True


def get_benchmark_config() -> BenchmarkConfig:
    """Get benchmark configuration"""
    return BenchmarkConfig()
