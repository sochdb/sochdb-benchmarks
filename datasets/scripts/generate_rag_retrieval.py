#!/usr/bin/env python3
"""
RAG Retrieval Dataset Generator

Generates synthetic RAG chunk data for benchmarking.
Each chunk has an embedding, doc_id, chunk_id, source, language, access_level, and created_at.
"""

import argparse
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

# Languages for multi-lingual RAG
LANGUAGES = ["en", "es", "fr", "de", "zh", "ja", "ko", "pt", "it", "ru"]
LANGUAGE_WEIGHTS = [0.5, 0.1, 0.08, 0.07, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02]

# Document sources
SOURCES = [
    "documentation", "wiki", "blog", "api_docs", "tutorials",
    "faq", "support_tickets", "internal_docs", "research_papers", "changelogs"
]

# Access levels (0 = public, higher = more restricted)
ACCESS_LEVELS = [0, 1, 2, 3, 4]
ACCESS_WEIGHTS = [0.6, 0.2, 0.1, 0.05, 0.05]


def generate_chunk_embedding(
    dim: int,
    doc_id: int,
    source: str,
    language: str,
    seed: int
) -> np.ndarray:
    """Generate a chunk embedding with document clustering."""
    rng = np.random.RandomState(seed)
    
    # Base random vector
    vec = rng.randn(dim).astype(np.float32)
    
    # Document bias (chunks from same doc cluster together)
    doc_bias = np.zeros(dim, dtype=np.float32)
    doc_bias_idx = (doc_id * 7) % dim
    doc_bias[doc_bias_idx:doc_bias_idx + dim // 10] = 0.3
    
    # Source bias
    source_idx = SOURCES.index(source) if source in SOURCES else 0
    source_bias = np.zeros(dim, dtype=np.float32)
    source_bias_idx = (source_idx * 13) % dim
    source_bias[source_bias_idx:source_bias_idx + dim // len(SOURCES)] = 0.2
    
    # Language bias
    lang_idx = LANGUAGES.index(language) if language in LANGUAGES else 0
    lang_bias = np.zeros(dim, dtype=np.float32)
    lang_bias_idx = (lang_idx * 17) % dim
    lang_bias[lang_bias_idx:lang_bias_idx + dim // len(LANGUAGES)] = 0.15
    
    vec = vec + doc_bias + source_bias + lang_bias
    
    # Normalize
    vec = vec / np.linalg.norm(vec)
    return vec


def generate_rag_chunks(
    num_chunks: int,
    dimension: int,
    chunks_per_doc: int = 20,
    seed: int = 42
) -> tuple:
    """
    Generate RAG chunk dataset.
    
    Returns:
        embeddings: np.ndarray of shape (num_chunks, dimension)
        metadata: list of dicts
    """
    embeddings = np.zeros((num_chunks, dimension), dtype=np.float32)
    metadata = []
    
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Base timestamp (1 year ago)
    base_time = datetime.now() - timedelta(days=365)
    
    num_docs = num_chunks // chunks_per_doc + 1
    
    idx = 0
    doc_id = 0
    chunk_in_doc = 0
    
    # Pre-assign document properties
    doc_sources = [rng.choice(SOURCES) for _ in range(num_docs)]
    doc_languages = rng.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=num_docs)
    doc_access = rng.choices(ACCESS_LEVELS, weights=ACCESS_WEIGHTS, k=num_docs)
    doc_times = [base_time + timedelta(days=rng.uniform(0, 365)) for _ in range(num_docs)]
    
    while idx < num_chunks:
        source = doc_sources[doc_id]
        language = doc_languages[doc_id]
        access_level = doc_access[doc_id]
        
        # Generate embedding
        embeddings[idx] = generate_chunk_embedding(
            dimension, doc_id, source, language, seed + idx
        )
        
        # Chunk timestamp slightly after doc creation
        chunk_time = doc_times[doc_id] + timedelta(seconds=chunk_in_doc * 60)
        
        metadata.append({
            "id": idx,
            "doc_id": doc_id,
            "chunk_id": chunk_in_doc,
            "source": source,
            "language": language,
            "access_level": access_level,
            "created_at": chunk_time.isoformat(),
        })
        
        idx += 1
        chunk_in_doc += 1
        
        if chunk_in_doc >= chunks_per_doc:
            doc_id += 1
            chunk_in_doc = 0
        
        if idx % 50000 == 0:
            print(f"  Generated {idx}/{num_chunks} chunks...")
    
    return embeddings, metadata


def generate_queries(
    num_queries: int,
    dimension: int,
    stratified: bool = True,
    ood_percent: float = 10.0,
    seed: int = 12345
) -> tuple:
    """
    Generate query embeddings with diverse filter selectivity.
    
    Returns:
        query_embeddings: np.ndarray
        query_metadata: list of dicts with filter parameters
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    queries = np.zeros((num_queries, dimension), dtype=np.float32)
    query_meta = []
    
    # Define selectivity tiers
    selectivity_tiers = ["tight", "medium", "broad"]
    
    for i in range(num_queries):
        # Determine if OOD query
        is_ood = rng.random() * 100 < ood_percent
        
        if is_ood:
            # Out-of-distribution: random noise
            queries[i] = np_rng.randn(dimension).astype(np.float32)
            selectivity = "ood"
        else:
            # In-distribution: similar to real chunks
            source = rng.choice(SOURCES)
            language = rng.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS)[0]
            queries[i] = generate_chunk_embedding(
                dimension, rng.randint(0, 1000), source, language, seed + i * 1000
            )
            selectivity = rng.choice(selectivity_tiers) if stratified else "medium"
        
        # Normalize
        queries[i] = queries[i] / np.linalg.norm(queries[i])
        
        # Generate filter parameters based on selectivity
        filter_lang = None
        filter_access = None
        filter_sources = None
        
        if selectivity == "tight":
            filter_lang = rng.choice(LANGUAGES)
            filter_access = rng.randint(0, 2)
            filter_sources = rng.sample(SOURCES, 2)
        elif selectivity == "medium":
            filter_lang = rng.choice(LANGUAGES)
            filter_access = rng.randint(0, 3)
        elif selectivity == "broad":
            filter_access = rng.randint(0, 4)
        # OOD queries have no filters
        
        query_meta.append({
            "id": i,
            "selectivity": selectivity,
            "filter_language": filter_lang,
            "filter_access_lte": filter_access,
            "filter_sources_in": filter_sources,
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
    batch_size = 100
    
    for batch_start in range(0, len(queries), batch_size):
        batch_end = min(batch_start + batch_size, len(queries))
        batch = queries[batch_start:batch_end]
        
        for i, query in enumerate(batch):
            global_idx = batch_start + i
            
            # Compute distances
            distances = np.linalg.norm(embeddings - query, axis=1)
            
            # Get top-k
            top_k_indices = np.argsort(distances)[:k]
            top_k_distances = distances[top_k_indices]
            
            truth[global_idx] = {
                "neighbors": top_k_indices.tolist(),
                "distances": top_k_distances.tolist(),
            }
        
        if (batch_end) % 1000 == 0:
            print(f"  Computed {batch_end}/{len(queries)} queries...")
    
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
    
    # Compute SHA256
    sha256 = hashlib.sha256()
    sha256.update(embeddings.tobytes())
    embeddings_hash = sha256.hexdigest()
    
    # Save meta.json
    meta = {
        "name": config["name"],
        "description": config["description"],
        "type": "rag_retrieval",
        "dimension": config["dimension"],
        "num_chunks": len(metadata),
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
    parser = argparse.ArgumentParser(description="Generate RAG retrieval dataset")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-n", "--chunks", type=int, default=200000, help="Number of chunks")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--queries", type=int, default=10000, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=50, help="K for ground truth")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    config = {
        "name": output_dir.name,
        "description": f"RAG retrieval dataset: {args.chunks} chunks, {args.dim}-dim",
        "num_chunks": args.chunks,
        "dimension": args.dim,
        "seed": args.seed,
        "k": args.k,
    }
    
    print(f"Generating RAG retrieval dataset:")
    print(f"  Chunks: {args.chunks}")
    print(f"  Dimension: {args.dim}")
    print(f"  Queries: {args.queries}")
    print()
    
    print("Generating RAG chunks...")
    embeddings, metadata = generate_rag_chunks(
        args.chunks, args.dim, seed=args.seed
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
