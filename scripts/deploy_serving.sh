set -euo pipefail

MODEL_ID="${1:-}"
SERVING_NAME="${SERVING_NAME:-sentiment-serving}"
ENDPOINT="${ENDPOINT:-sentiment}"
SERVING_ID="${SERVING_ID:-}"
CLEARML_SERVING_BIN="${CLEARML_SERVING_BIN:-clearml-serving}"

if ! command -v "${CLEARML_SERVING_BIN}" >/dev/null 2>&1 && [[ -x ".venv/bin/clearml-serving" ]]; then
  CLEARML_SERVING_BIN=".venv/bin/clearml-serving"
fi

if [[ -z "${MODEL_ID}" ]]; then
  echo "Usage: $0 <clearml_registry_model_id>"
  exit 1
fi

if [[ -z "${SERVING_ID}" ]]; then
  echo "Creating ClearML Serving service '${SERVING_NAME}'..."
  "${CLEARML_SERVING_BIN}" create --name "${SERVING_NAME}" || true
  echo
  echo "Find the serving service id with:"
  echo "  ${CLEARML_SERVING_BIN} list"
  echo
  echo "Then rerun this script with SERVING_ID, for example:"
  echo "  SERVING_ID=<SERVING_ID> $0 ${MODEL_ID}"
  exit 0
fi

"${CLEARML_SERVING_BIN}" --id "${SERVING_ID}" model add \
  --endpoint "${ENDPOINT}" \
  --model-id "${MODEL_ID}" \
  --engine sklearn \
  --preprocess serving/preprocess.py

echo "Added endpoint '${ENDPOINT}' for model '${MODEL_ID}' to serving service '${SERVING_ID}'."
echo "Launch inference locally on port 8090:"
echo "  CLEARML_SERVING_TASK_ID=${SERVING_ID} docker compose --profile serving up serving-inference"
