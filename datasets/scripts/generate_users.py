#!/usr/bin/env python3
"""
Generate synthetic user records for KV benchmarks.
Output format: JSONL (optionally compressed with zstd)
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path

def generate_users(n: int, output_path: Path, compress: bool = False):
    """Generate n synthetic user records."""
    
    print(f"Generating {n} user records...")
    
    lines = []
    for i in range(n):
        record = {
            "key": f"users/{i}",
            "value": {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "score": i % 100
            }
        }
        lines.append(json.dumps(record))
    
    content = "\n".join(lines)
    content_bytes = content.encode('utf-8')
    
    # Calculate hash
    sha256 = hashlib.sha256(content_bytes).hexdigest()
    
    output_file = output_path / "data.jsonl"
    
    if compress:
        try:
            import zstandard as zstd
            output_file = output_path / "data.jsonl.zst"
            cctx = zstd.ZstdCompressor(level=3)
            compressed = cctx.compress(content_bytes)
            with open(output_file, 'wb') as f:
                f.write(compressed)
            print(f"Wrote {len(compressed)} bytes (compressed) to {output_file}")
        except ImportError:
            print("zstandard not installed, writing uncompressed...")
            with open(output_file, 'w') as f:
                f.write(content)
            print(f"Wrote {len(content_bytes)} bytes to {output_file}")
    else:
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Wrote {len(content_bytes)} bytes to {output_file}")
    
    # Update meta.json
    meta_path = output_path / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['sha256'] = sha256
        meta['records'] = n
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Updated {meta_path} with sha256: {sha256[:16]}...")
    
    return sha256

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic user records')
    parser.add_argument('-n', '--count', type=int, default=100000, 
                       help='Number of records to generate')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--compress', action='store_true',
                       help='Compress output with zstd')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sha256 = generate_users(args.count, output_path, args.compress)
    print(f"\nDone! SHA256: {sha256}")

if __name__ == "__main__":
    main()
