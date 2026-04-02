#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash scripts/validate_existing_model_cpu.sh <Classifier> <ModelDir> <InputFile> [OutputDir]" >&2
  echo "Example: bash scripts/validate_existing_model_cpu.sh TransformerJobOffersClassifier /data/model tests/data/x_test.txt /tmp/job-ads-check" >&2
  exit 1
fi

CLASSIFIER="$1"
MODEL_DIR="$2"
INPUT_FILE="$3"
OUTPUT_DIR="${4:-/tmp/job-ads-classifier-check}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PRED_PATH="${OUTPUT_DIR}/predictions.txt"

mkdir -p "${OUTPUT_DIR}"

cd "${ROOT_DIR}"

echo "[1/4] CLI smoke check"
python main.py --help >/dev/null

echo "[2/4] Fast pytest suite"
pytest -m "not integration"

echo "[3/4] Existing model pytest check"
EXISTING_MODEL_DIR="${MODEL_DIR}" \
EXISTING_MODEL_INPUT="${INPUT_FILE}" \
EXISTING_MODEL_CLASSIFIER="${CLASSIFIER}" \
pytest tests/test_existing_model_cpu.py -q

echo "[4/4] Direct CPU prediction"
if [ "${CLASSIFIER}" = "TransformerJobOffersClassifier" ]; then
  python main.py predict "${CLASSIFIER}" \
    -x "${INPUT_FILE}" \
    -m "${MODEL_DIR}" \
    -p "${PRED_PATH}" \
    -A cpu \
    -P 32 \
    -T 1
else
  python main.py predict "${CLASSIFIER}" \
    -x "${INPUT_FILE}" \
    -m "${MODEL_DIR}" \
    -p "${PRED_PATH}" \
    -T 1
fi

echo
echo "Validation completed."
echo "Predictions: ${PRED_PATH}"
echo "Prediction labels map: ${PRED_PATH}.map"

