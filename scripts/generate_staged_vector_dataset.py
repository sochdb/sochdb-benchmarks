#!/usr/bin/env python3
"""
Generate a synthetic normalized vector dataset for staged scale benchmarks.

The output format matches benchmarks/run_vector_workload.py:
- embeddings.f32
- queries.f32
- meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalized_chunk(rng: np.random.Generator, rows: int, dim: int) -> np.ndarray:
    vectors = rng.standard_normal((rows, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid divide-by-zero if a row is all zeros.
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a staged synthetic vector dataset")
    parser.add_argument("--output-dir", required=True, help="Directory to write dataset files")
    parser.add_argument("--target-gib", type=float, default=10.0, help="Target embedding payload size in GiB")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--queries", type=int, default=1000, help="Number of query vectors")
    parser.add_argument("--chunk-vectors", type=int, default=50000, help="Vectors per write chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gib * (1024**3))
    bytes_per_vector = args.dim * 4
    num_vectors = target_bytes // bytes_per_vector
    actual_bytes = num_vectors * bytes_per_vector

    embeddings_path = output_dir / "embeddings.f32"
    queries_path = output_dir / "queries.f32"
    meta_path = output_dir / "meta.json"

    rng = np.random.default_rng(args.seed)

    with embeddings_path.open("wb") as fh:
        written = 0
        while written < num_vectors:
            chunk_rows = min(args.chunk_vectors, num_vectors - written)
            chunk = normalized_chunk(rng, chunk_rows, args.dim)
            chunk.astype(np.float32).tofile(fh)
            written += chunk_rows
            if written % max(args.chunk_vectors * 10, 1) == 0 or written == num_vectors:
                print(f"generated {written:,}/{num_vectors:,} vectors")

    queries = normalized_chunk(rng, args.queries, args.dim)
    queries.astype(np.float32).tofile(queries_path)

    meta = {
        "name": f"synthetic_{args.target_gib:g}gib_{args.dim}d",
        "source": "synthetic-normalized-gaussian",
        "dimension": args.dim,
        "num_vectors": num_vectors,
        "num_queries": args.queries,
        "target_gib": args.target_gib,
        "embedding_bytes": actual_bytes,
        "seed": args.seed,
        "files": {
            "embeddings": "embeddings.f32",
            "queries": "queries.f32",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
