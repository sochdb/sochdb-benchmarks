#!/usr/bin/env bash
set -euo pipefail

# Staged large-scale SochDB vector benchmark runner.
#
# This lane is for scale/throughput characterization, not retrieval-quality work.
# It uses a synthetic normalized-vector dataset sized by embedding payload.
#
# Optional inputs:
#   WORK_ROOT         default: $HOME/sochdb-benchmark-runs
#   SOCHDB_REPO       default: $HOME/sochdb
#   PYTHON_BIN        default: python3
#   TARGET_GIB        default: 10
#   DIM               default: 768
#   QUERIES           default: 1000
#   M                 default: 16
#   EF_CONSTRUCTION   default: 100
#   EF_SEARCH         default: 64
#   BATCH_SIZE        default: 1000
#
# Example:
#   TARGET_GIB=10 DIM=768 \
#   $HOME/sochdb-benchmark-runs/work/run_sochdb_stage_vector.sh

WORK_ROOT="${WORK_ROOT:-${HOME}/sochdb-benchmark-runs}"
SOCHDB_REPO="${SOCHDB_REPO:-${HOME}/sochdb}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VECTOR_RUNNER="${VECTOR_RUNNER:-${SCRIPT_DIR}/run_bulk_vector_workload.py}"
SOCHDB_PYTHONPATH="${SOCHDB_PYTHONPATH:-${HOME}/sochdb/sochdb-python/python}"
SOCHDB_LIB_PATH="${SOCHDB_LIB_PATH:-${HOME}/sochdb/target/release}"
SOCHDB_BULK_BIN_DIR="${SOCHDB_BULK_BIN_DIR:-${HOME}/test-venv/bin}"
TARGET_GIB="${TARGET_GIB:-10}"
DIM="${DIM:-768}"
QUERIES="${QUERIES:-250}"
M="${M:-16}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
EF_SEARCH="${EF_SEARCH:-64}"
BATCH_SIZE="${BATCH_SIZE:-1000}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_NAME="synthetic_${TARGET_GIB}gib_${DIM}d"
DATASET_DIR="${WORK_ROOT}/datasets/${DATASET_NAME}"
RUN_DIR="${WORK_ROOT}/results/${RUN_ID}_${DATASET_NAME}"
LOG_DIR="${WORK_ROOT}/logs"
LOG_FILE="${LOG_DIR}/sochdb-stage-${RUN_ID}.log"
RESULT_JSON="${RUN_DIR}/sochdb_vector.json"
METADATA_JSON="${RUN_DIR}/metadata.json"
INDEX_PATH="${RUN_DIR}/sochdb_vector.hnsw"

mkdir -p "${DATASET_DIR}" "${RUN_DIR}" "${LOG_DIR}"

{
  echo "=== SochDB Staged Vector Benchmark ==="
  echo "run_id=${RUN_ID}"
  echo "dataset_name=${DATASET_NAME}"
  echo "dataset_dir=${DATASET_DIR}"
  echo "result_json=${RESULT_JSON}"
  echo "target_gib=${TARGET_GIB}"
  echo "dim=${DIM}"
  echo "queries=${QUERIES}"
  echo "M=${M}"
  echo "ef_construction=${EF_CONSTRUCTION}"
  echo "ef_search=${EF_SEARCH}"
  echo "batch_size=${BATCH_SIZE}"
  echo "vector_runner=${VECTOR_RUNNER}"
  echo "sochdb_pythonpath=${SOCHDB_PYTHONPATH}"
  echo "sochdb_lib_path=${SOCHDB_LIB_PATH}"
  echo "sochdb_bulk_bin_dir=${SOCHDB_BULK_BIN_DIR}"
  echo
  echo "--- machine snapshot ---"
  hostname
  uname -a
  nproc
  free -h || true
  df -h /
  echo
} | tee "${LOG_FILE}"

if [[ ! -f "${DATASET_DIR}/meta.json" ]]; then
  echo "Generating staged dataset..." | tee -a "${LOG_FILE}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_staged_vector_dataset.py" \
    --output-dir "${DATASET_DIR}" \
    --target-gib "${TARGET_GIB}" \
    --dim "${DIM}" \
    --queries "${QUERIES}" | tee -a "${LOG_FILE}"
else
  echo "Using existing dataset at ${DATASET_DIR}" | tee -a "${LOG_FILE}"
fi

echo "Running SochDB vector workload..." | tee -a "${LOG_FILE}"
(
  cd "${SOCHDB_REPO}"
  env \
    PYTHONPATH="${SOCHDB_PYTHONPATH}" \
    SOCHDB_LIB_PATH="${SOCHDB_LIB_PATH}" \
    PATH="${SOCHDB_BULK_BIN_DIR}:${PATH}" \
    "${PYTHON_BIN}" "${VECTOR_RUNNER}" \
    --dataset "${DATASET_DIR}" \
    --queries "${QUERIES}" \
    --M "${M}" \
    --ef-construction "${EF_CONSTRUCTION}" \
    --ef-search "${EF_SEARCH}" \
    --batch-size "${BATCH_SIZE}" \
    --index-path "${INDEX_PATH}" \
    --output "${RESULT_JSON}"
) | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" - <<PY > "${METADATA_JSON}"
import json
from datetime import datetime, timezone
from pathlib import Path

dataset_dir = Path("${DATASET_DIR}")
meta = json.loads((dataset_dir / "meta.json").read_text())

payload = {
    "run_id": "${RUN_ID}",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_name": "${DATASET_NAME}",
    "dataset_dir": "<benchmark-workspace>/datasets/${DATASET_NAME}",
    "result_json": "<benchmark-workspace>/results/${RUN_ID}_${DATASET_NAME}/sochdb_vector.json",
    "index_path": "<benchmark-workspace>/results/${RUN_ID}_${DATASET_NAME}/sochdb_vector.hnsw",
    "target_gib": float("${TARGET_GIB}"),
    "dimension": int("${DIM}"),
    "queries": int("${QUERIES}"),
    "M": int("${M}"),
    "ef_construction": int("${EF_CONSTRUCTION}"),
    "ef_search": int("${EF_SEARCH}"),
    "batch_size": int("${BATCH_SIZE}"),
    "sochdb_bulk_bin_dir": "<benchmark-workspace-or-server-bin>/sochdb-bulk",
    "dataset_meta": meta,
}
print(json.dumps(payload, indent=2))
PY

echo "Saved result to ${RESULT_JSON}" | tee -a "${LOG_FILE}"
echo "Saved metadata to ${METADATA_JSON}" | tee -a "${LOG_FILE}"
