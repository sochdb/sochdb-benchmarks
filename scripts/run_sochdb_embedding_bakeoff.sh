#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATASET_DIR:-}" ]]; then
  echo "DATASET_DIR must be set" >&2
  exit 1
fi

SOCHDB_REPO="${SOCHDB_REPO:-${HOME}/sochdb}"
WORK_ROOT="${WORK_ROOT:-${HOME}/sochdb-benchmark-runs}"
PYTHON_BIN="${PYTHON_BIN:-${WORK_ROOT}/.venv/bin/python}"
EMBED_ROOT="${EMBED_ROOT:-${WORK_ROOT}/embeddings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-fastembed}"
MODELS="${MODELS:-BAAI/bge-small-en-v1.5,thenlper/gte-small}"
MODEL_BACKENDS="${MODEL_BACKENDS:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${EMBED_ROOT}" "${RUN_DIR}"

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
IFS=',' read -r -a MODEL_BACKEND_LIST <<< "${MODEL_BACKENDS}"

resolve_backend() {
  local model="$1"
  local entry

  for entry in "${MODEL_BACKEND_LIST[@]}"; do
    if [[ -z "${entry}" ]]; then
      continue
    fi
    if [[ "${entry%%=*}" == "${model}" ]]; then
      echo "${entry#*=}"
      return 0
    fi
  done

  echo "${EMBEDDING_BACKEND}"
}

for model in "${MODEL_LIST[@]}"; do
  safe_name="$(echo "${model}" | tr '/:' '__')"
  embedding_dir="${EMBED_ROOT}/${safe_name}"
  result_dir="${RUN_DIR}/${safe_name}"
  backend="$(resolve_backend "${model}")"
  mkdir -p "${result_dir}"

  echo "Generating embeddings for model=${model} backend=${backend}"
  "${PYTHON_BIN}" "${SOCHDB_REPO}/benchmarks/retrieval/embed.py" \
    --backend "${backend}" \
    --model "${model}" \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${embedding_dir}"

  echo "Running quality sweep for model=${model}"
  DATASET_DIR="${DATASET_DIR}" \
  EMBEDDING_DIR="${embedding_dir}" \
  OUTPUT_ROOT="${result_dir}" \
  RUN_ID="sweep" \
  SOCHDB_REPO="${SOCHDB_REPO}" \
  "${WORK_ROOT}/work/run_sochdb_grpc_quality_sweep.sh"
done

echo "Saved embedding bakeoff results to ${RUN_DIR}"
