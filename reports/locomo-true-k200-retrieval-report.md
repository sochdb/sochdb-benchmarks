# LoCoMo Retrieval-Only True K200 Run Report

## Summary

One full LoCoMo retrieval-only run with final output depth K=200, scored at K=20, K=50, K=100, K=150, and K=200.

This is **not** five separate runs. It is one true top-200 ranked retrieval output, re-scored at progressively shallower cutoffs.

**Source path:** `benchmarks/paper/locomo/results/full_single_metadata_sochdb_nvidia_dim2048_true_k200/retrieval.jsonl`

## Output Depth Sanity

| Rows | min_len | max_len | avg_len | Distribution          |
|-----:|--------:|--------:|--------:|-----------------------|
| 1986 |      200 |      200 |  200.00 | (200, 1986)           |

Every row contains exactly 200 `retrieved_memory_ids`.

## Overall Metrics by K

| K  | Hit@K  | Recall@K |
|---:|-------:|---------:|
|  20 | 0.7678 | 0.7122 |
|  50 | 0.8503 | 0.7969 |
| 100 | 0.9191 | 0.8756 |
| 150 | 0.9570 | 0.9255 |
| 200 | 0.9717 | 0.9484 |

## Category Metrics

### K=100

| Category    | Hit@100 | Recall@100 |
|-------------|--------:|-----------:|
| adversarial |  0.9126 |     0.9081 |
| multi_hop   |  0.7416 |     0.6508 |
| open_domain |  0.9310 |     0.9257 |
| single_hop  |  0.9253 |     0.6995 |
| temporal    |  0.9406 |     0.9159 |

### K=200

| Category    | Hit@200 | Recall@200 |
|-------------|--------:|-----------:|
| adversarial |  0.9619 |     0.9596 |
| multi_hop   |  0.8652 |     0.7622 |
| open_domain |  0.9834 |     0.9806 |
| single_hop  |  0.9858 |     0.8726 |
| temporal    |  0.9719 |     0.9667 |

## Key Observations

- **Overall retrieval ceiling is strong.** K=200 reaches 97.17 Hit and 94.84 Recall across all 1,986 questions.
- **K=100 is not enough.** 91.91 Hit@100 and 87.56 Recall@100 leaves significant evidence between ranks 101–200.
- **K=150 crosses 90/90.** 95.70 Hit and 92.55 Recall at K=150.
- **Top-100 error is largely a ranking/compression problem, not candidate absence.** Many gold evidence memories exist in ranks 101–200, confirming they were retrieved but not in the top 100.
- **Multi-hop is the weakest category at every cutoff.**
  - K=100: 74.16 Hit / 65.08 Recall
  - K=200: 86.52 Hit / 76.22 Recall
  - Even at K=200, multi-hop Recall stays below 77%, indicating structural retrieval gaps.

## Next-Step Recommendations

1. **Ranking compression:** Investigate re-ranking or score-based pruning to push ranks 101–200 evidence into the top 100.
2. **Multi-hop:** Source-aware multiview retrieval, entity-constrained probes, local neighbor expansion, and coverage-aware evidence-set selection.
3. **Single-hop:** Narrowing the Recall gap (87.26 at K=200) likely requires better short-span exact fact extraction.
4. **Further K-depth experiments** are low priority; the ceiling is now well-characterized.

## Validation

```bash
python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py <retrieval.jsonl>
```

The report references an existing retrieval output path from the source repo and records row-length sanity checks showing all 1,986 rows contain 200 retrieved IDs.

## Non-Goals

- No benchmark code changes.
- No large JSONL outputs committed.
- No final answer-quality claims.
- No API keys or secrets.
- No claim that multi-hop is solved.