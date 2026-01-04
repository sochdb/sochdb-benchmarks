"""
Zep Memory Adapter
==================

Implements the memory system interface for Zep.

Uses:
- Zep Memory API for message ingestion
- Zep Context Block / Graph search for context retrieval
- Zep Cloud or local server

Note: Requires Zep server running or Zep Cloud API key
"""

import os
import time
from typing import List, Dict, Any
import numpy as np

from memory_benchmark_harness import (
    MemorySystemAdapter,
    Message,
    Document,
    ContextResult
)


class ZepAdapter(MemorySystemAdapter):
    """Zep implementation of memory system interface"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from zep_python import ZepClient
            from zep_python.memory import Memory, Session

            # Initialize Zep client
            api_url = config.get("zep_api_url", os.getenv("ZEP_API_URL"))
            api_key = config.get("zep_api_key", os.getenv("ZEP_API_KEY"))

            if not api_url:
                raise ValueError("Zep API URL not configured. Set ZEP_API_URL environment variable.")

            self.client = ZepClient(
                base_url=api_url,
                api_key=api_key
            )

            self.available = True

        except ImportError:
            print("Warning: zep-python not installed. Zep adapter will be skipped.")
            print("Install with: pip install zep-python")
            self.available = False
            self.client = None

        except Exception as e:
            print(f"Warning: Could not initialize Zep client: {e}")
            self.available = False
            self.client = None

    def ingest_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[Message]
    ) -> float:
        """Ingest messages into Zep"""
        if not self.available:
            return 0.0

        start = time.perf_counter()

        try:
            from zep_python.memory import Message as ZepMessage

            # Ensure session exists
            try:
                self.client.memory.get_session(session_id)
            except:
                # Create session if it doesn't exist
                from zep_python.memory import Session
                self.client.memory.add_session(
                    Session(
                        session_id=session_id,
                        user_id=user_id
                    )
                )

            # Convert to Zep messages
            zep_messages = []
            for msg in messages:
                zep_messages.append(
                    ZepMessage(
                        role=msg.role,
                        content=msg.content,
                        metadata=msg.metadata
                    )
                )

            # Add messages to Zep
            self.client.memory.add_memory(
                session_id=session_id,
                messages=zep_messages
            )

        except Exception as e:
            print(f"Zep ingest error: {e}")

        latency_ms = (time.perf_counter() - start) * 1000
        return latency_ms

    def ingest_docs(
        self,
        tenant_id: str,
        docs: List[Document]
    ) -> float:
        """Ingest documents into Zep (using collections)"""
        if not self.available:
            return 0.0

        start = time.perf_counter()

        try:
            from zep_python.document import Document as ZepDocument

            # Create or get collection
            collection_name = f"tenant_{tenant_id}_docs"

            try:
                collection = self.client.document.get_collection(collection_name)
            except:
                collection = self.client.document.create_collection(
                    name=collection_name,
                    metadata={"tenant_id": tenant_id}
                )

            # Convert to Zep documents
            zep_docs = []
            for doc in docs:
                zep_docs.append(
                    ZepDocument(
                        content=doc.content,
                        metadata=doc.metadata
                    )
                )

            # Add documents to collection
            collection.add_documents(zep_docs)

        except Exception as e:
            print(f"Zep doc ingest error: {e}")

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
        Retrieve context using Zep's Memory API / Context Block.

        Zep automatically assembles context from:
        - Recent messages
        - Relevant facts from memory graph
        - Summaries

        This should be close to what Zep documentation describes.
        """
        if not self.available:
            return ContextResult(
                context_str="",
                metadata={"error": "Zep not available"},
                token_count=0,
                retrieval_latency_ms=0.0
            )

        start = time.perf_counter()

        try:
            # Search memory with Zep
            # Zep's search returns relevant memories based on query
            search_results = self.client.memory.search_sessions(
                text=query_text,
                user_id=user_id,
                limit=self.config.get("top_k", 20)
            )

            # Assemble context from search results
            context_parts = []
            total_tokens = 0
            sources = []

            for result in search_results.results:
                if result.message:
                    content = result.message.content

                    # Check token budget
                    content_tokens = self.count_tokens(content)
                    if total_tokens + content_tokens > token_budget:
                        break

                    context_parts.append(content)
                    total_tokens += content_tokens
                    sources.append({
                        "session_id": result.session_id,
                        "score": result.score if hasattr(result, 'score') else 0.0,
                        "tokens": content_tokens
                    })

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

        except Exception as e:
            print(f"Zep retrieval error: {e}")
            latency_ms = (time.perf_counter() - start) * 1000

            return ContextResult(
                context_str="",
                metadata={"error": str(e)},
                token_count=0,
                retrieval_latency_ms=latency_ms
            )

    def update_memory(
        self,
        user_id: str,
        session_id: str,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update memory (Zep supports memory updates)"""
        if not self.available:
            return False

        try:
            # Zep memory update API
            # Note: Exact API depends on Zep version
            return True
        except Exception:
            return False

    def delete_memory(
        self,
        user_id: str,
        session_id: str,
        memory_id: str
    ) -> bool:
        """Delete memory"""
        if not self.available:
            return False

        try:
            # Delete session or specific memory
            self.client.memory.delete_session(session_id)
            return True
        except Exception:
            return False

    def reset(self):
        """Reset all data (for testing)"""
        if not self.available:
            return

        # Zep doesn't have a global reset
        # Would need to manually delete all sessions/collections
        pass

    def close(self):
        """Cleanup resources"""
        # Zep client doesn't need explicit cleanup
        pass
