"""
ToonDB Memory Adapter
====================

Implements the memory system interface for ToonDB.

Uses:
- Context Query for context assembly within token budget
- HNSW index for O(log n) retrieval
- Hierarchical storage for messages and docs
"""

import os
import time
import json
from typing import List, Dict, Any
import numpy as np
from openai import AzureOpenAI

from memory_benchmark_harness import (
    MemorySystemAdapter,
    Message,
    Document,
    ContextResult
)


class ToonDBAdapter(MemorySystemAdapter):
    """ToonDB implementation of memory system interface"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        from toondb import Database, VectorIndex

        # Initialize ToonDB database
        db_path = config.get("db_path", "/tmp/toondb_benchmark.db")
        self.db = Database.open(db_path)

        # Initialize HNSW index for vector search
        self.embedding_dim = config.get("embedding_dim", 1536)
        self.hnsw_index = VectorIndex(
            dimension=self.embedding_dim,
            max_connections=config.get("hnsw_m", 16),
            ef_construction=config.get("hnsw_ef_construction", 100)
        )

        # Initialize Azure OpenAI for embeddings
        self.embedder = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

        # Track memory IDs
        self.next_id = 0
        self.id_to_key = {}  # Maps HNSW IDs to storage keys

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        response = self.embedder.embeddings.create(
            model=self.embedding_deployment,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _get_message_key(self, user_id: str, session_id: str, msg_idx: int) -> str:
        """Generate hierarchical key for message"""
        return f"user.{user_id}.session.{session_id}.msg.{msg_idx}"

    def _get_doc_key(self, tenant_id: str, doc_idx: int) -> str:
        """Generate hierarchical key for document"""
        return f"tenant.{tenant_id}.doc.{doc_idx}"

    def ingest_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[Message]
    ) -> float:
        """Ingest messages into ToonDB"""
        start = time.perf_counter()

        for i, msg in enumerate(messages):
            # Generate embedding
            embedding = self._embed(msg.content)

            # Store message metadata
            key = self._get_message_key(user_id, session_id, i)
            metadata = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "user_id": user_id,
                "session_id": session_id,
                **msg.metadata
            }

            self.db.put(
                f"{key}.metadata".encode(),
                json.dumps(metadata).encode()
            )

            # Store embedding
            self.db.put(
                f"{key}.embedding".encode(),
                embedding.tobytes()
            )

            # Add to HNSW index
            hnsw_id = self.next_id
            self.next_id += 1
            self.id_to_key[hnsw_id] = key

            self.hnsw_index.insert_batch(
                np.array([hnsw_id], dtype=np.uint64),
                embedding.reshape(1, -1)
            )

        latency_ms = (time.perf_counter() - start) * 1000
        return latency_ms

    def ingest_docs(
        self,
        tenant_id: str,
        docs: List[Document]
    ) -> float:
        """Ingest documents into ToonDB"""
        start = time.perf_counter()

        for i, doc in enumerate(docs):
            # Generate embedding
            embedding = self._embed(doc.content)

            # Store document metadata
            key = self._get_doc_key(tenant_id, i)
            metadata = {
                "content": doc.content,
                "tenant_id": tenant_id,
                **doc.metadata
            }

            self.db.put(
                f"{key}.metadata".encode(),
                json.dumps(metadata).encode()
            )

            # Store embedding
            self.db.put(
                f"{key}.embedding".encode(),
                embedding.tobytes()
            )

            # Add to HNSW index
            hnsw_id = self.next_id
            self.next_id += 1
            self.id_to_key[hnsw_id] = key

            self.hnsw_index.insert_batch(
                np.array([hnsw_id], dtype=np.uint64),
                embedding.reshape(1, -1)
            )

        latency_ms = (time.perf_counter() - start) * 1000
        return latency_ms

    def retrieve_context(
        self,
        user_id: str,
        session_id: str,
        query_text: str,
        token_budget: int
    ) -> ContextResult:
        """
        Retrieve context using ToonDB's HNSW search + token-aware assembly.

        Strategy:
        1. Embed query
        2. HNSW search for top-k relevant memories
        3. Assemble context within token budget
        4. Return context string
        """
        start = time.perf_counter()

        # Embed query
        query_embedding = self._embed(query_text)

        # Search HNSW index (over-fetch for filtering)
        top_k = self.config.get("top_k", 20)
        results = self.hnsw_index.search(query_embedding, k=top_k)

        # Retrieve and assemble context within token budget
        context_parts = []
        total_tokens = 0
        sources = []

        for hnsw_id, score in results:
            key = self.id_to_key.get(int(hnsw_id))
            if not key:
                continue

            # Filter by user/session if this is a message
            if "user." in key:
                # Check if it belongs to this user/session
                if f"user.{user_id}.session.{session_id}" not in key:
                    continue

            # Load metadata
            metadata_bytes = self.db.get(f"{key}.metadata".encode())
            if not metadata_bytes:
                continue

            metadata = json.loads(metadata_bytes.decode())
            content = metadata.get("content", "")

            # Check token budget
            content_tokens = self.count_tokens(content)
            if total_tokens + content_tokens > token_budget:
                break  # Budget exceeded

            context_parts.append(content)
            total_tokens += content_tokens
            sources.append({
                "key": key,
                "score": float(score),
                "tokens": content_tokens
            })

        # Assemble final context
        context_str = "\n\n".join(context_parts)

        latency_ms = (time.perf_counter() - start) * 1000

        return ContextResult(
            context_str=context_str,
            metadata={
                "num_sources": len(sources),
                "budget_used": total_tokens,
                "budget_limit": token_budget
            },
            token_count=total_tokens,
            retrieval_latency_ms=latency_ms,
            sources=sources
        )

    def update_memory(
        self,
        user_id: str,
        session_id: str,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update memory metadata"""
        try:
            key = f"user.{user_id}.session.{session_id}.msg.{memory_id}"

            # Load existing metadata
            metadata_bytes = self.db.get(f"{key}.metadata".encode())
            if not metadata_bytes:
                return False

            metadata = json.loads(metadata_bytes.decode())
            metadata.update(updates)

            # Update metadata
            self.db.put(
                f"{key}.metadata".encode(),
                json.dumps(metadata).encode()
            )

            return True
        except Exception:
            return False

    def delete_memory(
        self,
        user_id: str,
        session_id: str,
        memory_id: str
    ) -> bool:
        """Delete memory (mark as deleted)"""
        try:
            key = f"user.{user_id}.session.{session_id}.msg.{memory_id}"

            # Mark as deleted in metadata
            metadata_bytes = self.db.get(f"{key}.metadata".encode())
            if not metadata_bytes:
                return False

            metadata = json.loads(metadata_bytes.decode())
            metadata["deleted"] = True

            self.db.put(
                f"{key}.metadata".encode(),
                json.dumps(metadata).encode()
            )

            return True
        except Exception:
            return False

    def reset(self):
        """Reset all data"""
        # Close and reopen DB (clears in-memory state)
        self.db.close()
        db_path = self.config.get("db_path", "/tmp/toondb_benchmark.db")

        # Remove old DB file or directory
        import os
        import shutil
        if os.path.exists(db_path):
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
            else:
                os.remove(db_path)

        # Recreate
        from toondb import Database, VectorIndex
        self.db = Database.open(db_path)

        # Reset HNSW index
        self.hnsw_index = VectorIndex(
            dimension=self.embedding_dim,
            max_connections=self.config.get("hnsw_m", 16),
            ef_construction=self.config.get("hnsw_ef_construction", 100)
        )

        self.next_id = 0
        self.id_to_key = {}

    def close(self):
        """Cleanup resources"""
        if self.db:
            self.db.close()
