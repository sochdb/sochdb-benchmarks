# LoCoMo Configurable Retrieval Depth

## Summary

`--candidate-k` controls internal candidate depth. `--k` controls final emitted retrieval depth.

## Validation

```bash
uv run python benchmarks/paper/locomo/runners/test_hybrid_locomo_retrieval.py
```

## Non-Goals

- No retrieval algorithm change.
- No scoring change.
- No answer-quality claim.
- No large output committed.
