#!/usr/bin/env python3
"""
Integrated Memory Macrobenchmark: "The Agent Loop"
mimics a live agent conversation where system must simultaneously:
1. Write: Append new observations (KV + Vector).
2. Read: Vector search with hard metadata filter (e.g., timestamp > T).
3. Assemble: Fetch context to build a prompt.

Measures End-to-End Latency (P99) for this complete cycle.
"""

import time
import uuid
import random
import numpy as np
import os
import shutil
import sqlite3
import argparse
from typing import List, Dict, Tuple

# Optional imports for baselines
try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import toondb
except ImportError:
    toondb = None

# =============================================================================
# CONFIGURATION
# =============================================================================
DIMENSION = 128  # Keep small for synthetic speed, or 1536 for realistic
NUM_TURNS = 100
INITIAL_MEMORY_SIZE = 5000
FILTER_RATIO = 0.2  # Search only recent 20%
TOP_K = 5

class BaseAgentMemory:
    def add_observation(self, text: str, vector: List[float], metadata: Dict):
        raise NotImplementedError

    def recall(self, query: List[float], min_timestamp: int) -> List[str]:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

# =============================================================================
# TOONDB IMPLEMENTATION (Unified)
# =============================================================================
class ToonDBMemory(BaseAgentMemory):
    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):
            shutil.rmtree(path)
        
        self.db = toondb.Database.open(path)
        self.ns = self.db.get_or_create_namespace("agent_007")
        self.collection = self.ns.create_collection("episodic_memory", dimension=DIMENSION)

    def add_observation(self, text: str, vector: List[float], metadata: Dict):
        # ID is a timestamp-based or uuid
        obs_id = str(metadata['timestamp']) + "_" + str(uuid.uuid4())[:8]
        # In ToonDB, we store content and metadata directly with the vector
        self.collection.insert(
            id=obs_id,
            vector=vector,
            metadata=metadata,
            content=text
        )

    def recall(self, query: List[float], min_timestamp: int) -> List[str]:
        # Filter: "timestamp" > min_timestamp
        # ToonDB supports metadata filtering
        results = self.collection.vector_search(
            vector=query,
            k=TOP_K,
            filter={"timestamp": {"$gt": min_timestamp}}
        )
        # Assemble context directly from results
        return [r.content for r in results]

    def name(self) -> str:
        return "ToonDB (Unified)"

    def cleanup(self):
        # No explicit close needed for embedded usually, but good practice
        pass

# =============================================================================
# BASELINE: CHROMA + SQLITE (Fragmented)
# =============================================================================
class FragmentedMemory(BaseAgentMemory):
    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        
        # SQLite for "Source of Truth" / KV
        self.sql_conn = sqlite3.connect(os.path.join(path, "memory.db"))
        self.sql_conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                content TEXT,
                timestamp INTEGER,
                metadata TEXT
            )
        """)
        self.sql_conn.commit()
        
        # Chroma for Vector Index
        self.chroma = chromadb.PersistentClient(path=os.path.join(path, "chroma"))
        self.collection = self.chroma.create_collection("episodic_mem")

    def add_observation(self, text: str, vector: List[float], metadata: Dict):
        obs_id = str(metadata['timestamp']) + "_" + str(uuid.uuid4())[:8]
        
        # 1. Write to SQLite (Atomicity gap here in real world, but ignoring for bench)
        self.sql_conn.execute(
            "INSERT INTO observations (id, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (obs_id, text, metadata['timestamp'], str(metadata))
        )
        self.sql_conn.commit()
        
        # 2. Write to Chroma
        self.collection.add(
            ids=[obs_id],
            embeddings=[vector],
            metadatas=[metadata]
        )

    def recall(self, query: List[float], min_timestamp: int) -> List[str]:
        # 1. Query Chroma (with filter support)
        # Chroma supports where={"timestamp": {"$gt": val}}
        
        results = self.collection.query(
            query_embeddings=[query],
            n_results=TOP_K,
            where={"timestamp": {"$gt": min_timestamp}}
        )
        
        ids = results['ids'][0]
        
        # 2. Fetch full content from SQLite (Simulating "Assemble")
        # In this simple case Chroma stores metadata, but often full documents are in DB
        # To fairly compare "Fragmented" approach where DB is SOT:
        placeholders = ','.join('?' for _ in ids)
        rows = self.sql_conn.execute(
            f"SELECT content FROM observations WHERE id IN ({placeholders})",
            ids
        ).fetchall()
        
        return [r[0] for r in rows]

    def name(self) -> str:
        return "SQLite + Chroma (Fragmented)"
    
    def cleanup(self):
        self.sql_conn.close()

# =============================================================================
# WORKLOAD GENERATOR
# =============================================================================
def run_benchmark(memory_system: BaseAgentMemory):
    metrics = {
        "write_latencies": [],
        "read_latencies": []
    }
    
    # Pre-populate
    print(f"[{memory_system.name()}] Pre-populating {INITIAL_MEMORY_SIZE} memories...")
    start_ts = int(time.time()) - 100000
    
    for i in range(INITIAL_MEMORY_SIZE):
        ts = start_ts + i * 10
        vec = np.random.randn(DIMENSION).tolist()
        memory_system.add_observation(
            text=f"Observation {i} about the world state.",
            vector=vec,
            metadata={"timestamp": ts, "type": "observation"}
        )
    
    print(f"[{memory_system.name()}] Starting Agent Loop ({NUM_TURNS} turns)...")
    
    current_ts = start_ts + INITIAL_MEMORY_SIZE * 10
    
    for _ in range(NUM_TURNS):
        # SIMULATE TURN
        
        # 1. "Write": User says something, agent stores it
        current_ts += 10
        w_vec = np.random.randn(DIMENSION).tolist()
        
        t0 = time.perf_counter()
        memory_system.add_observation(
            text=f"Turn {_} observation",
            vector=w_vec,
            metadata={"timestamp": current_ts, "type": "dialogue"}
        )
        metrics["write_latencies"].append((time.perf_counter() - t0) * 1000)
        
        # 2. "Read": Retrieve relevant context from last 24h (simulated)
        # Filter window: last 20% of timeline
        lookback = int((current_ts - start_ts) * FILTER_RATIO)
        min_ts = current_ts - lookback
        q_vec = np.random.randn(DIMENSION).tolist()
        
        t0 = time.perf_counter()
        ctx = memory_system.recall(q_vec, min_timestamp=min_ts)
        # Simulate generic processing of context
        _ = len(str(ctx)) 
        metrics["read_latencies"].append((time.perf_counter() - t0) * 1000)
        
    return metrics

def print_stats(name, latencies):
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"  {name:<10}: P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")
    return p99

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    
    results = {}
    
    # Run ToonDB
    if toondb:
        try:
            mem = ToonDBMemory("/tmp/bench_toondb_macro")
            metrics = run_benchmark(mem)
            results["toondb"] = metrics
            mem.cleanup()
            shutil.rmtree("/tmp/bench_toondb_macro", ignore_errors=True)
        except Exception as e:
            print(f"ToonDB failed: {e}")
            import traceback
            traceback.print_exc()

    # Run Fragmented
    if chromadb:
        try:
            mem = FragmentedMemory("/tmp/bench_frag_macro")
            metrics = run_benchmark(mem)
            results["fragmented"] = metrics
            mem.cleanup()
            shutil.rmtree("/tmp/bench_frag_macro", ignore_errors=True)
        except Exception as e:
            print(f"Baseline failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("MACRO-BENCHMARK RESULTS (End-to-End Latency)")
    print("="*60)
    
    for sys_name in ["toondb", "fragmented"]:
        if sys_name in results:
            m = results[sys_name]
            print(f"\nSystem: {sys_name.upper()}")
            print_stats("Write", m["write_latencies"])
            print_stats("Read+Assemble", m["read_latencies"])

