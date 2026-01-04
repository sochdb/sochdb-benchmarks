#!/usr/bin/env python3
"""
Generate random unit vectors for vector search benchmarks.
Output format: Binary f32 arrays
"""

import argparse
import json
import struct
import hashlib
from pathlib import Path
import numpy as np

def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate n random unit vectors of dimension dim."""
    np.random.seed(seed)
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Compute ground truth nearest neighbors using brute-force cosine similarity."""
    ground_truth = []
    for i, query in enumerate(queries):
        # Cosine similarity (vectors are normalized, so dot product = cosine sim)
        similarities = np.dot(vectors, query)
        # Get top-k indices (highest similarity = nearest)
        top_k = np.argsort(similarities)[-k:][::-1]
        ground_truth.append({
            "query_id": i,
            "neighbors": top_k.tolist(),
            "similarities": similarities[top_k].tolist()
        })
    return ground_truth

def main():
    parser = argparse.ArgumentParser(description='Generate random unit vectors')
    parser.add_argument('-n', '--count', type=int, default=10000,
                       help='Number of vectors to generate')
    parser.add_argument('-d', '--dimension', type=int, default=128,
                       help='Vector dimension')
    parser.add_argument('-q', '--queries', type=int, default=100,
                       help='Number of query vectors')
    parser.add_argument('-k', '--topk', type=int, default=10,
                       help='Number of neighbors for ground truth')
    parser.add_argument('-s', '--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output directory path')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.count} vectors with dimension {args.dimension}...")
    
    # Generate vectors
    vectors = generate_vectors(args.count, args.dimension, args.seed)
    
    # Generate queries (different seed)
    print(f"Generating {args.queries} query vectors...")
    queries = generate_vectors(args.queries, args.dimension, args.seed + 1000)
    
    # Write vectors as binary f32
    vectors_file = output_path / "vectors.f32"
    with open(vectors_file, 'wb') as f:
        # Header: n, dim as uint32
        f.write(struct.pack('<II', args.count, args.dimension))
        # Data
        f.write(vectors.tobytes())
    print(f"Wrote vectors to {vectors_file} ({vectors_file.stat().st_size} bytes)")
    
    # Write queries as binary f32
    queries_file = output_path / "queries.f32"
    with open(queries_file, 'wb') as f:
        # Header: n, dim as uint32
        f.write(struct.pack('<II', args.queries, args.dimension))
        # Data
        f.write(queries.tobytes())
    print(f"Wrote queries to {queries_file} ({queries_file.stat().st_size} bytes)")
    
    # Compute ground truth
    print(f"Computing ground truth (top-{args.topk})...")
    ground_truth = compute_ground_truth(vectors, queries, args.topk)
    
    gt_file = output_path / "ground_truth.json"
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote ground truth to {gt_file}")
    
    # Calculate combined hash
    hasher = hashlib.sha256()
    hasher.update(vectors.tobytes())
    hasher.update(queries.tobytes())
    sha256 = hasher.hexdigest()
    
    # Update meta.json
    meta_path = output_path / "meta.json"
    meta = {
        "name": f"vectors_{args.count // 1000}k_{args.dimension}",
        "description": f"{args.count} random unit vectors with {args.dimension} dimensions",
        "type": "vector",
        "dimension": args.dimension,
        "vectors": args.count,
        "queries": args.queries,
        "format": "f32_binary",
        "files": {
            "vectors": "vectors.f32",
            "queries": "queries.f32",
            "ground_truth": "ground_truth.json"
        },
        "ground_truth": {
            "k": args.topk,
            "method": "brute_force_cosine"
        },
        "seed": args.seed,
        "sha256": sha256
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")
    
    print(f"\nDone! SHA256: {sha256}")

if __name__ == "__main__":
    main()
