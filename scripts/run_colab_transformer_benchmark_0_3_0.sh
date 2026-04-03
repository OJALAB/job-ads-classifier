#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/OJALAB/job-ads-classifier.git}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-/content/job-ads-classifier}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SERVER="${SERVER:-https://repod.icm.edu.pl}"
DATASET_PID="${DATASET_PID:-doi:10.18150/OCUTSI}"
MODEL_FILENAME="${MODEL_FILENAME:-transformer-bottom-base-2024.zip}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/content/repod-models}"
DOWNLOAD_DIR="${DOWNLOAD_ROOT}/downloads"
EXTRACT_DIR="${DOWNLOAD_ROOT}/extracted"

BIG_INPUT_PATH="${BIG_INPUT_PATH:-/content/x_test_big.txt}"
INPUT_MULTIPLIER="${INPUT_MULTIPLIER:-200}"
PRED_PATH_PREFIX="${PRED_PATH_PREFIX:-/content/job-ads-transformer-gpu-bench}"

ACCELERATOR="${ACCELERATOR:-gpu}"
PRECISION="${PRECISION:-16-mixed}"
THREADS="${THREADS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REPEATS="${REPEATS:-3}"

echo "[INFO] Colab transformer benchmark 0.3.0"
echo "[INFO] Repo ref: ${REPO_REF}"
echo "[INFO] Model file: ${MODEL_FILENAME}"
echo "[INFO] Accelerator: ${ACCELERATOR}"
echo "[INFO] Precision: ${PRECISION}"
echo "[INFO] Batch size: ${BATCH_SIZE}"
echo "[INFO] Repeats: ${REPEATS}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[INFO] Cloning repository into ${REPO_DIR}"
  rm -rf "${REPO_DIR}"
  git clone --branch "${REPO_REF}" "${REPO_URL}" "${REPO_DIR}"
else
  echo "[INFO] Reusing existing repository checkout in ${REPO_DIR}"
  git -C "${REPO_DIR}" fetch --tags origin
  git -C "${REPO_DIR}" checkout "${REPO_REF}"
  git -C "${REPO_DIR}" pull --ff-only origin "${REPO_REF}"
fi

cd "${REPO_DIR}"

echo "[INFO] Installing Python dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements-colab-0.3.0.txt
"${PYTHON_BIN}" -m pip install -e .

echo "[INFO] Verifying torch and runtime"
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_DIR}"

echo "[INFO] Downloading and extracting ${MODEL_FILENAME} from RepOD"
MODEL_DIR="$(
"${PYTHON_BIN}" - "${SERVER}" "${DATASET_PID}" "${MODEL_FILENAME}" "${DOWNLOAD_DIR}" "${EXTRACT_DIR}" <<'PY'
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

server, dataset_pid, filename, download_dir, extract_dir = sys.argv[1:]
download_dir = Path(download_dir)
extract_dir = Path(extract_dir)
download_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

query = urllib.parse.urlencode({"persistentId": dataset_pid})
metadata_url = f"{server}/api/datasets/:persistentId/versions/:latest-published?{query}"
with urllib.request.urlopen(metadata_url) as response:
    metadata = json.load(response)

files = metadata.get("data", {}).get("files", [])
selected = None
for entry in files:
    data_file = entry.get("dataFile", {})
    if data_file.get("filename") == filename:
        selected = data_file
        break

if selected is None:
    raise SystemExit(f"Could not find {filename} in RepOD metadata")

file_id = selected["id"]
archive_path = download_dir / filename
if not archive_path.exists():
    download_url = f"{server}/api/access/datafile/{file_id}"
    print(f"[PY] Downloading {download_url}", file=sys.stderr)
    urllib.request.urlretrieve(download_url, archive_path)
else:
    print(f"[PY] Reusing existing archive {archive_path}", file=sys.stderr)

extract_root = extract_dir / archive_path.stem
if not extract_root.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)

    nested_dir = extract_root / archive_path.stem
    if nested_dir.is_dir():
        for child in nested_dir.iterdir():
            target = extract_root / child.name
            if target.exists():
                continue
            child.rename(target)
        nested_dir.rmdir()
else:
    print(f"[PY] Reusing extracted directory {extract_root}", file=sys.stderr)

candidates = []
for path in [extract_root, *extract_root.rglob("*")]:
    if path.is_dir() and (path / "transformer_arch.bin").exists():
        candidates.append(path)

if not candidates:
    for path in [extract_root, *extract_root.rglob("*")]:
        if path.is_dir() and (path / "ckpts").exists():
            candidates.append(path)

if not candidates:
    raise SystemExit(f"Could not infer model dir under {extract_root}")

candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
print(candidates[0])
PY
)"

echo "[INFO] MODEL_DIR=${MODEL_DIR}"

echo "[INFO] Building a larger benchmark input file"
"${PYTHON_BIN}" - "${REPO_DIR}" "${BIG_INPUT_PATH}" "${INPUT_MULTIPLIER}" <<'PY'
import sys
from pathlib import Path

repo_dir, output_path, multiplier = sys.argv[1], sys.argv[2], int(sys.argv[3])
source_path = Path(repo_dir) / "tests/data/x_test.txt"
dest_path = Path(output_path)
lines = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expanded = lines * multiplier
dest_path.write_text("\n".join(expanded) + "\n", encoding="utf-8")
print(f"{dest_path} {len(expanded)}")
PY

echo "[INFO] Running lazy vs batched benchmark on GPU"
PYTHON_BIN="${PYTHON_BIN}" \
"${PYTHON_BIN}" scripts/benchmark_0_3_0.py existing-model-compare \
  --classifier TransformerJobOffersClassifier \
  --model-dir "${MODEL_DIR}" \
  --input-file "${BIG_INPUT_PATH}" \
  --pred-path-prefix "${PRED_PATH_PREFIX}" \
  --accelerator "${ACCELERATOR}" \
  --precision "${PRECISION}" \
  --threads "${THREADS}" \
  --batch-size "${BATCH_SIZE}" \
  --repeats "${REPEATS}"
