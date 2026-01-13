#!/usr/bin/env python3
"""
Hardware Microbenchmark: SIMD Efficiency
Measures the raw throughput of the cosine distance kernel.

Goal: Validate that SochDB's Rust-based FFI kernels are using AVX/NEON 
instructions effectively by comparing against highly optimized NumPy (BLAS/LAPACK).

If SochDB is comparable to or faster than NumPy for small-vector ops, 
it proves the overhead of FFI is negligible and SIMD is active.
"""

import time
import numpy as np
import sys

try:
    import sochdb
except ImportError:
    print("SochDB not installed.")
    sys.exit(0)

# =============================================================================
# CONFIG
# =============================================================================
DIM = 128
ITERATIONS = 100_000
WARMUP = 10_000

def bench_numpy(v1, v2, iters):
    start = time.perf_counter()
    for _ in range(iters):
        # Cosine distance = 1 - dot(norm(v1), norm(v2))
        # Assuming v1, v2 pre-normalized for fair comparison of dot product
        np.dot(v1, v2)
    return time.perf_counter() - start

def bench_sochdb(index, v1, iters):
    # SochDB doesn't expose raw distance function in Python API usually.
    # We simulate it by doing a search for k=1 on an index with 1 vector.
    # This includes HNSW protocol overhead, so it's a "strict" test.
    # If this is close to NumPy, the kernel is blazing fast.
    
    start = time.perf_counter()
    for _ in range(iters):
        index.search(v1, k=1)
    return time.perf_counter() - start

def main():
    print("="*60)
    print(f"HARDWARE SENSITIVITY TEST (NEON/AVX Kernel)")
    print(f"Vector Dim: {DIM}, Iterations: {ITERATIONS}")
    print("="*60)
    
    # Setup Data
    v1 = np.random.randn(DIM).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    
    # Setup SochDB
    # Create an index with just 1 vector to isolate distance calc/overhead
    try:
        index = sochdb.VectorIndex(dimension=DIM, max_connections=16, ef_construction=100)
        index.insert(0, v1)
    except Exception as e:
        print(f"Failed to init SochDB: {e}")
        return

    # Warmup
    print("Warming up...")
    bench_numpy(v1, v1, WARMUP)
    bench_sochdb(index, v1, WARMUP)
    
    # Run NumPy
    print(f"Testing NumPy (BLAS)...", end="", flush=True)
    t_np = bench_numpy(v1, v1, ITERATIONS)
    ops_np = ITERATIONS / t_np
    print(f" {t_np:.4f}s ({ops_np:,.0f} ops/sec)")
    
    # Run SochDB
    print(f"Testing SochDB (Rust FFI + HNSW)...", end="", flush=True)
    t_toon = bench_sochdb(index, v1, ITERATIONS)
    ops_toon = ITERATIONS / t_toon
    print(f" {t_toon:.4f}s ({ops_toon:,.0f} ops/sec)")
    
    # Analysis
    ratio = ops_toon / ops_np
    print("-" * 60)
    print(f"Relative Performance: {ratio:.2f}x vs NumPy")
    
    if ratio > 0.5:
        print("✅ PASS: SochDB kernel is efficient (includes FFI overhead)")
    else:
        print("⚠️ WARNING: Significant FFI/Kernel bottleneck detected")

if __name__ == "__main__":
    main()
