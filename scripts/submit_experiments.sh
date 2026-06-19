set -euo pipefail

DATASET_ID="${1:-}"
QUEUE="${CLEARML_QUEUE:-students}"
PYTHON="${PYTHON:-.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-outputs/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-outputs/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ -z "${DATASET_ID}" ]]; then
  echo "Usage: $0 <clearml_dataset_id>"
  exit 1
fi

"${PYTHON}" -m src.train \
  --dataset-id "${DATASET_ID}" \
  --execute-remotely \
  --queue "${QUEUE}" \
  --experiment-name sentiment-logreg-baseline \
  --classifier logreg \
  --c 1.0 \
  --ngram-max 1

"${PYTHON}" -m src.train \
  --dataset-id "${DATASET_ID}" \
  --execute-remotely \
  --queue "${QUEUE}" \
  --experiment-name sentiment-linearsvc-bigrams \
  --classifier linearsvc \
  --c 0.7 \
  --ngram-max 2
