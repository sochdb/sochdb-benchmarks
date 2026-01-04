#!/usr/bin/env python3
"""
Agent Memory Dataset Generator

Generates synthetic agent memory data for benchmarking.
Each memory has an embedding, timestamp, agent_id, topic_tags, importance, and text_len.
"""

import argparse
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random
import struct

# Topic categories for realistic memory content
TOPICS = [
    "task", "conversation", "observation", "learning", "decision",
    "user_preference", "system_event", "error", "success", "reminder"
]

def generate_memory_embedding(dim: int, topic: str, seed: int) -> np.ndarray:
    """Generate a memory embedding with topic-based clustering."""
    rng = np.random.RandomState(seed)
    
    # Base random vector
    vec = rng.randn(dim).astype(np.float32)
    
    # Add topic-specific bias for clustering
    topic_idx = TOPICS.index(topic) if topic in TOPICS else 0
    bias = np.zeros(dim, dtype=np.float32)
    bias_start = (topic_idx * dim // len(TOPICS)) % dim
    bias[bias_start:bias_start + dim // len(TOPICS)] = 0.5
    
    vec = vec + bias
    
    # Normalize
    vec = vec / np.linalg.norm(vec)
    return vec


def generate_agent_memories(
    num_agents: int,
    memories_per_agent: int,
    dimension: int,
    seed: int = 42
) -> tuple:
    """
    Generate agent memory dataset.
    
    Returns:
        embeddings: np.ndarray of shape (total, dimension)
        metadata: list of dicts with {timestamp, agent_id, topic_tags, importance, text_len}
    """
    total = num_agents * memories_per_agent
    embeddings = np.zeros((total, dimension), dtype=np.float32)
    metadata = []
    
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Base timestamp (30 days ago)
    base_time = datetime.now() - timedelta(days=30)
    
    idx = 0
    for agent_id in range(num_agents):
        # Agent-specific random state for consistency
        agent_seed = seed + agent_id
        
        for mem_idx in range(memories_per_agent):
            # Select topic(s)
            num_topics = rng.randint(1, 3)
            topics = rng.sample(TOPICS, num_topics)
            
            # Generate embedding
            embeddings[idx] = generate_memory_embedding(
                dimension, topics[0], agent_seed + mem_idx
            )
            
            # Generate timestamp (more recent memories are more common)
            # Exponential distribution biased toward recent
            days_ago = np_rng.exponential(7)  # Mean of 7 days ago
            days_ago = min(days_ago, 30)  # Cap at 30 days
            timestamp = (base_time + timedelta(days=30 - days_ago)).isoformat()
            
            # Importance score (0.0 to 1.0)
            importance = np_rng.beta(2, 5)  # Skewed toward lower values
            
            # Text length (characters)
            text_len = int(np_rng.lognormal(5, 1))  # ~150 chars median
            text_len = max(10, min(text_len, 10000))
            
            metadata.append({
                "id": idx,
                "agent_id": agent_id,
                "timestamp": timestamp,
                "topic_tags": topics,
                "importance": float(importance),
                "text_len": text_len,
            })
            
            idx += 1
            
        if (agent_id + 1) % 100 == 0:
            print(f"  Generated memories for {agent_id + 1}/{num_agents} agents...")
    
    return embeddings, metadata


def generate_queries(
    num_queries: int,
    dimension: int,
    include_near_duplicates: bool = True,
    include_topic_shifts: bool = True,
    seed: int = 12345
) -> tuple:
    """
    Generate query embeddings with diverse characteristics.
    
    Returns:
        query_embeddings: np.ndarray
        query_metadata: list of dicts
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    queries = np.zeros((num_queries, dimension), dtype=np.float32)
    query_meta = []
    
    for i in range(num_queries):
        query_type = "normal"
        
        if include_near_duplicates and i % 10 == 0:
            # Near-duplicate: very similar to previous
            if i > 0:
                queries[i] = queries[i-1] + np_rng.randn(dimension).astype(np.float32) * 0.01
                query_type = "near_duplicate"
            else:
                queries[i] = np_rng.randn(dimension).astype(np.float32)
        elif include_topic_shifts and i % 7 == 0:
            # Topic shift: sudden change in query direction
            topic = rng.choice(TOPICS)
            queries[i] = generate_memory_embedding(dimension, topic, seed + i * 1000)
            query_type = "topic_shift"
        else:
            # Normal random query
            queries[i] = np_rng.randn(dimension).astype(np.float32)
        
        # Normalize
        queries[i] = queries[i] / np.linalg.norm(queries[i])
        
        query_meta.append({
            "id": i,
            "type": query_type,
            "filter_agent_id": rng.randint(0, 99) if rng.random() < 0.8 else None,
            "filter_importance_gte": rng.uniform(0.3, 0.7) if rng.random() < 0.3 else None,
        })
    
    return queries, query_meta


def compute_ground_truth(
    embeddings: np.ndarray,
    queries: np.ndarray,
    k: int = 50
) -> dict:
    """Compute exact brute-force nearest neighbors."""
    print(f"Computing ground truth for {len(queries)} queries (k={k})...")
    
    truth = {}
    
    for i, query in enumerate(queries):
        # Compute distances to all embeddings
        distances = np.linalg.norm(embeddings - query, axis=1)
        
        # Get top-k indices
        top_k_indices = np.argsort(distances)[:k]
        top_k_distances = distances[top_k_indices]
        
        truth[i] = {
            "neighbors": top_k_indices.tolist(),
            "distances": top_k_distances.tolist(),
        }
        
        if (i + 1) % 500 == 0:
            print(f"  Computed {i + 1}/{len(queries)} queries...")
    
    return truth


def save_dataset(
    output_dir: Path,
    embeddings: np.ndarray,
    metadata: list,
    queries: np.ndarray,
    query_metadata: list,
    ground_truth: dict,
    config: dict
):
    """Save dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_dir / "embeddings.f32"
    embeddings.tofile(embeddings_path)
    print(f"Saved embeddings to {embeddings_path} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")
    
    # Save queries
    queries_path = output_dir / "queries.f32"
    queries.tofile(queries_path)
    print(f"Saved queries to {queries_path}")
    
    # Save query metadata
    query_meta_path = output_dir / "query_metadata.json"
    with open(query_meta_path, 'w') as f:
        json.dump(query_metadata, f)
    
    # Save ground truth
    truth_path = output_dir / "ground_truth.json"
    truth_file = {
        "version": "1.0",
        "dataset": config["name"],
        "k": config["k"],
        "num_queries": len(query_metadata),
        "queries": ground_truth,
    }
    with open(truth_path, 'w') as f:
        json.dump(truth_file, f)
    print(f"Saved ground truth to {truth_path}")
    
    # Compute SHA256 of embeddings
    sha256 = hashlib.sha256()
    sha256.update(embeddings.tobytes())
    embeddings_hash = sha256.hexdigest()
    
    # Save meta.json
    meta = {
        "name": config["name"],
        "description": config["description"],
        "type": "agent_memory",
        "dimension": config["dimension"],
        "num_agents": config["num_agents"],
        "memories_per_agent": config["memories_per_agent"],
        "total_vectors": len(metadata),
        "num_queries": len(query_metadata),
        "format": "f32_binary",
        "files": {
            "embeddings": "embeddings.f32",
            "metadata": "metadata.json",
            "queries": "queries.f32",
            "query_metadata": "query_metadata.json",
            "ground_truth": "ground_truth.json",
        },
        "generated_at": datetime.now().isoformat(),
        "seed": config["seed"],
        "sha256": embeddings_hash,
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta.json with SHA256: {embeddings_hash[:16]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate agent memory dataset")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--memories", type=int, default=1000, help="Memories per agent")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--queries", type=int, default=5000, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=50, help="K for ground truth")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    config = {
        "name": output_dir.name,
        "description": f"Agent memory dataset: {args.agents} agents × {args.memories} memories",
        "num_agents": args.agents,
        "memories_per_agent": args.memories,
        "dimension": args.dim,
        "seed": args.seed,
        "k": args.k,
    }
    
    print(f"Generating agent memory dataset:")
    print(f"  Agents: {args.agents}")
    print(f"  Memories per agent: {args.memories}")
    print(f"  Total vectors: {args.agents * args.memories}")
    print(f"  Dimension: {args.dim}")
    print(f"  Queries: {args.queries}")
    print()
    
    # Generate data
    print("Generating agent memories...")
    embeddings, metadata = generate_agent_memories(
        args.agents, args.memories, args.dim, args.seed
    )
    
    print(f"\nGenerating {args.queries} queries...")
    queries, query_metadata = generate_queries(
        args.queries, args.dim, seed=args.seed + 1000
    )
    
    print(f"\nComputing ground truth (k={args.k})...")
    ground_truth = compute_ground_truth(embeddings, queries, args.k)
    
    print(f"\nSaving dataset to {output_dir}...")
    save_dataset(
        output_dir, embeddings, metadata,
        queries, query_metadata, ground_truth, config
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
