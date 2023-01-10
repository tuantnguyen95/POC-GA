# Activate the environment first
TF_SERVING_HOST="${KOBITON_AI_TF_SERVING_HOST:-0.0.0.0}"
#TF_SERVING_HOST="3.115.126.43"
TF_SERVING_PORT="${KOBITON_AI_TF_SERVING_PORT:-3000}"
SERVICE_HOST="${KOBITON_AI_SERVICE_HOST:-0.0.0.0}"
SERVICE_PORT="${KOBITON_AI_SERVICE_PORT:-5000}"
export KOBITON_CONSUL_HOST="${KOBITON_CONSUL_HOST:-0.0.0.0}"
export KOBITON_CONSUL_REST_API_PORT="${KOBITON_CONSUL_REST_API_PORT:-8500}"
cd src && python -m service.app $TF_SERVING_HOST $TF_SERVING_PORT \
$SERVICE_HOST $SERVICE_PORT