#!/usr/bin/env bash
set -euo pipefail

# Staged pilot runner for the SochDB gRPC retrieval benchmark.
#
# Required inputs:
#   DATASET_DIR     directory with corpus.jsonl and queries.jsonl
#   EMBEDDING_DIR   directory with doc/query embedding outputs
#
# Optional inputs:
#   SOCHDB_GRPC_HOST   default: studio.agentslab.host
#   SOCHDB_GRPC_PORT   default: 50053
#   TOP_K              default: 5
#   INDEX_NAME_PREFIX  default: pilot
#   WORK_ROOT          default: $HOME/sochdb-benchmark-runs
#   SOCHDB_REPO        default: $HOME/sochdb
#   OUTPUT_ROOT        default: $WORK_ROOT/results
#
# Example:
#   DATASET_DIR=$HOME/sochdb-benchmark-runs/datasets/scifact \
#   EMBEDDING_DIR=$HOME/sochdb-benchmark-runs/datasets/scifact-embeddings \
#   $HOME/sochdb-benchmark-runs/work/run_sochdb_grpc_pilot.sh

SOCHDB_GRPC_HOST="${SOCHDB_GRPC_HOST:-studio.agentslab.host}"
SOCHDB_GRPC_PORT="${SOCHDB_GRPC_PORT:-50053}"
TOP_K="${TOP_K:-5}"
INDEX_NAME_PREFIX="${INDEX_NAME_PREFIX:-pilot}"
WORK_ROOT="${WORK_ROOT:-${HOME}/sochdb-benchmark-runs}"
SOCHDB_REPO="${SOCHDB_REPO:-${HOME}/sochdb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

: "${DATASET_DIR:?DATASET_DIR is required}"
: "${EMBEDDING_DIR:?EMBEDDING_DIR is required}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

RESULT_JSON="${RUN_DIR}/sochdb_grpc.json"
METADATA_JSON="${RUN_DIR}/metadata.json"
LOG_FILE="${LOG_DIR}/sochdb-grpc-pilot-${RUN_ID}.log"
INDEX_NAME="${INDEX_NAME_PREFIX}_${RUN_ID}"

{
  echo "=== SochDB gRPC Pilot Benchmark ==="
  echo "run_id=${RUN_ID}"
  echo "host=${SOCHDB_GRPC_HOST}"
  echo "port=${SOCHDB_GRPC_PORT}"
  echo "dataset_dir=${DATASET_DIR}"
  echo "embedding_dir=${EMBEDDING_DIR}"
  echo "result_json=${RESULT_JSON}"
  echo "index_name=${INDEX_NAME}"
  echo
  echo "--- machine snapshot ---"
  hostname
  uname -a
  nproc
  free -h || true
  df -h /
  echo
} | tee "${LOG_FILE}"

python3 "${SOCHDB_REPO}/benchmarks/retrieval/run_sochdb_grpc.py" \
  --host "${SOCHDB_GRPC_HOST}" \
  --port "${SOCHDB_GRPC_PORT}" \
  --dataset-dir "${DATASET_DIR}" \
  --embedding-dir "${EMBEDDING_DIR}" \
  --output "${RESULT_JSON}" \
  --k "${TOP_K}" \
  --index-name "${INDEX_NAME}" | tee -a "${LOG_FILE}"

python3 - <<PY > "${METADATA_JSON}"
import json
from datetime import datetime, timezone
metadata = {
    "run_id": "${RUN_ID}",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "grpc_host": "${SOCHDB_GRPC_HOST}",
    "grpc_port": int("${SOCHDB_GRPC_PORT}"),
    "dataset_dir": "${DATASET_DIR}",
    "embedding_dir": "${EMBEDDING_DIR}",
    "result_json": "${RESULT_JSON}",
    "index_name": "${INDEX_NAME}",
    "top_k": int("${TOP_K}"),
}
print(json.dumps(metadata, indent=2))
PY

echo "Saved result to ${RESULT_JSON}" | tee -a "${LOG_FILE}"
echo "Saved metadata to ${METADATA_JSON}" | tee -a "${LOG_FILE}"
