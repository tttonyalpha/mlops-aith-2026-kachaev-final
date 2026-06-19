set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-outputs/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-outputs/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

"${PYTHON}" -m src.train \
  --local-only \
  --experiment-name local-smoke-logreg \
  --classifier logreg \
  --ngram-max 1 \
  --output-dir outputs

"${PYTHON}" -m src.predict_local \
  --model-path outputs/local-smoke-logreg.joblib \
  --text "The interface is clean and reliable"
