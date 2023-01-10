# Activate the environment first
TF_SERVING_HOST="${KOBITON_AI_TF_SERVING_HOST:-0.0.0.0}"
TF_SERVING_PORT="${KOBITON_AI_TF_SERVING_PORT:-8500}"
SERVICE_HOST="${KOBITON_AI_SERVICE_HOST:-0.0.0.0}"
SERVICE_PORT="${KOBITON_AI_SERVICE_PORT:-5000}"

cd build && python -m service.benchmark.benchmark_tfserving $TF_SERVING_HOST $TF_SERVING_PORT 10

# Uncomment the following line for the ai service benchmarking
# python -m service.benchmark.benchmark_ai_service http://${SERVICE_HOST}:${SERVICE_PORT}/element_finding?session_id=1234 10