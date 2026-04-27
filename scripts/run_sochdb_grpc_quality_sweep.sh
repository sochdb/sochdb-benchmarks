#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATASET_DIR:-}" || -z "${EMBEDDING_DIR:-}" ]]; then
  echo "DATASET_DIR and EMBEDDING_DIR must be set" >&2
  exit 1
fi

SOCHDB_REPO="${SOCHDB_REPO:-/root/sochdb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/sochdb-benchmark-runs/results}"
HOST="${SOCHDB_GRPC_HOST:-studio.agentslab.host}"
PORT="${SOCHDB_GRPC_PORT:-50053}"
TOP_K="${TOP_K:-5}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

CONFIGS=(
  "fast 16 100 64"
  "balanced 32 150 96"
  "quality 48 200 128"
)

RESULT_FILES=()

for config in "${CONFIGS[@]}"; do
  read -r name m efc efs <<<"${config}"
  output_file="${RUN_DIR}/sochdb_grpc_${name}.json"
  echo "Running config=${name} m=${m} ef_construction=${efc} ef_search=${efs}"
  python3 "${SOCHDB_REPO}/benchmarks/retrieval/run_sochdb_grpc.py" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dataset-dir "${DATASET_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --k "${TOP_K}" \
    --m "${m}" \
    --ef-construction "${efc}" \
    --ef-search "${efs}" \
    --index-name "sweep_${name}_${RUN_ID}" \
    --output "${output_file}"
  RESULT_FILES+=("${output_file}")
done

python3 "${SOCHDB_REPO}/benchmarks/retrieval/evaluate.py" \
  --k "${TOP_K}" \
  --output-json "${RUN_DIR}/summary.json" \
  "${RESULT_FILES[@]}" | tee "${RUN_DIR}/summary.txt"

echo "Saved sweep results to ${RUN_DIR}"
