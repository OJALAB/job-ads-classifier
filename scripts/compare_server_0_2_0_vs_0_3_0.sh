#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

REF_OLD="${REF_OLD:-v0.2.0}"
REF_NEW="${REF_NEW:-HEAD}"
CLASSIFIER="${CLASSIFIER:-TransformerJobOffersClassifier}"
MODEL_DIR="${MODEL_DIR:-}"
INPUT_FILE="${INPUT_FILE:-tests/data/x_test.txt}"
ACCELERATOR="${ACCELERATOR:-cpu}"
PRECISION="${PRECISION:-32}"
THREADS="${THREADS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REPEATS="${REPEATS:-3}"
COMPARE_NEW_LAZY="${COMPARE_NEW_LAZY:-1}"
KEEP_WORKTREES="${KEEP_WORKTREES:-0}"
WORK_DIR="${WORK_DIR:-/tmp/job-ads-compare-work-$$}"
OUT_DIR="${OUT_DIR:-/tmp/job-ads-compare-results-$(date +%Y%m%d-%H%M%S)}"

usage() {
  cat <<EOF
Compare prediction runtime for ${REF_OLD} vs ${REF_NEW} on the same extracted model.

Required:
  --model-dir PATH         Extracted model directory to benchmark.

Optional:
  --classifier NAME        LinearJobOffersClassifier or TransformerJobOffersClassifier
  --input-file PATH        Input text file. Relative paths are resolved inside each checkout.
  --accelerator NAME       cpu by default
  --precision VALUE        32 by default
  --threads N              1 by default
  --batch-size N           64 by default
  --repeats N              3 by default
  --ref-old REF            v0.2.0 by default
  --ref-new REF            HEAD by default
  --out-dir PATH           Output directory for logs, predictions, and summaries
  --work-dir PATH          Temporary git worktree directory
  --skip-new-lazy          Skip the extra 0.3.0 lazy-mode measurement
  --keep-worktrees         Leave temporary worktrees on disk after the run

Examples:
  bash scripts/compare_server_0_2_0_vs_0_3_0.sh \\
    --classifier TransformerJobOffersClassifier \\
    --model-dir /opt/job-ads-classifier/models/transformer-bottom-base-2024/transformer-bottom-base-2024

  bash scripts/compare_server_0_2_0_vs_0_3_0.sh \\
    --classifier LinearJobOffersClassifier \\
    --model-dir /opt/job-ads-classifier/models/linear-bottom-2024/linear-bottom-2024 \\
    --threads 30
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --classifier)
      CLASSIFIER="$2"
      shift 2
      ;;
    --input-file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --ref-old)
      REF_OLD="$2"
      shift 2
      ;;
    --ref-new)
      REF_NEW="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --skip-new-lazy)
      COMPARE_NEW_LAZY=0
      shift
      ;;
    --keep-worktrees)
      KEEP_WORKTREES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_DIR}" ]]; then
  echo "--model-dir is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${OUT_DIR}"

OLD_TREE="${WORK_DIR}/old"
NEW_TREE="${WORK_DIR}/new"

cleanup() {
  if [[ "${KEEP_WORKTREES}" == "1" ]]; then
    return
  fi

  git -C "${REPO_DIR}" worktree remove --force "${OLD_TREE}" >/dev/null 2>&1 || true
  git -C "${REPO_DIR}" worktree remove --force "${NEW_TREE}" >/dev/null 2>&1 || true
  rm -rf "${WORK_DIR}"
}

trap cleanup EXIT

resolve_input_path() {
  local worktree="$1"
  if [[ "${INPUT_FILE}" = /* ]]; then
    printf '%s\n' "${INPUT_FILE}"
  else
    printf '%s\n' "${worktree}/${INPUT_FILE}"
  fi
}

prepare_worktree() {
  local target_dir="$1"
  local ref="$2"

  if [[ -d "${target_dir}" ]]; then
    git -C "${REPO_DIR}" worktree remove --force "${target_dir}" >/dev/null 2>&1 || true
  fi

  git -C "${REPO_DIR}" worktree add --detach "${target_dir}" "${ref}" >/dev/null
}

measure_predict() {
  local label="$1"
  local worktree="$2"
  shift 2
  local extra_args=("$@")
  local input_path
  local times_path="${OUT_DIR}/${label}.times.tsv"
  local last_pred=""

  input_path="$(resolve_input_path "${worktree}")"
  : > "${times_path}"

  for repeat in $(seq 1 "${REPEATS}"); do
    local pred_path="${OUT_DIR}/${label}.run${repeat}.pred.txt"
    local log_path="${OUT_DIR}/${label}.run${repeat}.log.txt"
    local time_path="${OUT_DIR}/${label}.run${repeat}.time.txt"
    local cmd=(
      "${PYTHON_BIN}" "${worktree}/main.py"
      predict "${CLASSIFIER}"
      -x "${input_path}"
      -m "${MODEL_DIR}"
      -p "${pred_path}"
      -A "${ACCELERATOR}"
      -P "${PRECISION}"
      -T "${THREADS}"
      -b "${BATCH_SIZE}"
    )

    if [[ "${#extra_args[@]}" -gt 0 ]]; then
      cmd+=("${extra_args[@]}")
    fi

    echo "[RUN] ${label} repeat ${repeat}: ${cmd[*]}" >&2
    /usr/bin/time -p -o "${time_path}" "${cmd[@]}" > "${log_path}" 2>&1
    awk '/^real / {print '"${repeat}"'\t"$2}' "${time_path}" >> "${times_path}"
    last_pred="${pred_path}"
  done

  printf '%s\n' "${last_pred}"
}

prepare_worktree "${OLD_TREE}" "${REF_OLD}"
prepare_worktree "${NEW_TREE}" "${REF_NEW}"

echo "[INFO] Output directory: ${OUT_DIR}"
echo "[INFO] Old ref: ${REF_OLD}"
echo "[INFO] New ref: ${REF_NEW}"
echo "[INFO] Classifier: ${CLASSIFIER}"
echo "[INFO] Model dir: ${MODEL_DIR}"
echo "[INFO] Input file: ${INPUT_FILE}"

OLD_PRED="$(measure_predict old_ref "${OLD_TREE}")"
NEW_BATCHED_PRED="$(measure_predict new_batched "${NEW_TREE}" --tokenization-mode batched)"
NEW_LAZY_PRED=""

if [[ "${CLASSIFIER}" == "TransformerJobOffersClassifier" && "${COMPARE_NEW_LAZY}" == "1" ]]; then
  NEW_LAZY_PRED="$(measure_predict new_lazy "${NEW_TREE}" --tokenization-mode lazy)"
fi

export OUT_DIR OLD_PRED NEW_BATCHED_PRED NEW_LAZY_PRED CLASSIFIER REF_OLD REF_NEW INPUT_FILE MODEL_DIR REPEATS
SUMMARY_PATH="${OUT_DIR}/summary.json"
MARKDOWN_PATH="${OUT_DIR}/summary.md"
export SUMMARY_PATH MARKDOWN_PATH

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np


def read_times(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            repeat, seconds = line.strip().split("\t")
            rows.append(float(seconds))
    return rows


def summarize_times(path):
    runs = read_times(path)
    return {
        "runs": runs,
        "avg_seconds": sum(runs) / len(runs),
        "min_seconds": min(runs),
        "max_seconds": max(runs),
    }


def load_prediction(path):
    data = np.loadtxt(path)
    return np.atleast_2d(data)


def compare_predictions(path_a, path_b):
    if not path_a or not path_b:
        return None

    pred_a = load_prediction(path_a)
    pred_b = load_prediction(path_b)
    map_a = Path(f"{path_a}.map").read_text(encoding="utf-8").splitlines()
    map_b = Path(f"{path_b}.map").read_text(encoding="utf-8").splitlines()

    return {
        "shape_a": list(pred_a.shape),
        "shape_b": list(pred_b.shape),
        "max_abs_diff": float(np.max(np.abs(pred_a - pred_b))),
        "mean_abs_diff": float(np.mean(np.abs(pred_a - pred_b))),
        "maps_equal": map_a == map_b,
    }


out_dir = Path(os.environ["OUT_DIR"])
summary = {
    "classifier": os.environ["CLASSIFIER"],
    "refs": {
        "old": os.environ["REF_OLD"],
        "new": os.environ["REF_NEW"],
    },
    "input_file": os.environ["INPUT_FILE"],
    "model_dir": os.environ["MODEL_DIR"],
    "repeats": int(os.environ["REPEATS"]),
    "results": {
        "old_ref": summarize_times(out_dir / "old_ref.times.tsv"),
        "new_batched": summarize_times(out_dir / "new_batched.times.tsv"),
    },
    "prediction_checks": {
        "old_vs_new_batched": compare_predictions(os.environ["OLD_PRED"], os.environ["NEW_BATCHED_PRED"]),
    },
}

if os.environ.get("NEW_LAZY_PRED"):
    summary["results"]["new_lazy"] = summarize_times(out_dir / "new_lazy.times.tsv")
    summary["prediction_checks"]["old_vs_new_lazy"] = compare_predictions(os.environ["OLD_PRED"], os.environ["NEW_LAZY_PRED"])
    summary["prediction_checks"]["new_lazy_vs_new_batched"] = compare_predictions(os.environ["NEW_LAZY_PRED"], os.environ["NEW_BATCHED_PRED"])
    summary["speedups"] = {
        "old_to_new_batched": summary["results"]["old_ref"]["avg_seconds"] / summary["results"]["new_batched"]["avg_seconds"],
        "new_lazy_to_new_batched": summary["results"]["new_lazy"]["avg_seconds"] / summary["results"]["new_batched"]["avg_seconds"],
    }
else:
    summary["speedups"] = {
        "old_to_new_batched": summary["results"]["old_ref"]["avg_seconds"] / summary["results"]["new_batched"]["avg_seconds"],
    }

summary_path = Path(os.environ["SUMMARY_PATH"])
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Version comparison",
    "",
    f"- classifier: `{summary['classifier']}`",
    f"- old ref: `{summary['refs']['old']}`",
    f"- new ref: `{summary['refs']['new']}`",
    f"- input: `{summary['input_file']}`",
    f"- model dir: `{summary['model_dir']}`",
    "",
    "## Timings",
    "",
]

for label, metrics in summary["results"].items():
    lines.append(f"- `{label}`: avg `{metrics['avg_seconds']:.4f}s`, min `{metrics['min_seconds']:.4f}s`, max `{metrics['max_seconds']:.4f}s`")

lines.extend([
    "",
    "## Speedups",
    "",
])
for label, value in summary["speedups"].items():
    lines.append(f"- `{label}`: `{value:.4f}x`")

lines.extend([
    "",
    "## Prediction checks",
    "",
])
for label, metrics in summary["prediction_checks"].items():
    if metrics is None:
        continue
    lines.append(
        f"- `{label}`: max_abs_diff `{metrics['max_abs_diff']:.10f}`, "
        f"mean_abs_diff `{metrics['mean_abs_diff']:.10f}`, maps_equal `{metrics['maps_equal']}`"
    )

markdown_path = Path(os.environ["MARKDOWN_PATH"])
markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\n[OK] JSON summary: {summary_path}")
print(f"[OK] Markdown summary: {markdown_path}")
PY
