#!/usr/bin/env bash
# Sweep request concurrency against an OpenAI-compatible endpoint (e.g. the Dynamo frontend)
# with AIPerf, and keep every run's artifacts for later comparison.
#
#   kubectl port-forward svc/qwen3-disagg-frontend 8000:8000 &
#   MODEL=Qwen/Qwen3-32B inference/bench/aiperf-sweep.sh
#
# Defaults model a chat/RAG-style load: 2k tokens in, 256 out. ignore_eos makes every request
# produce exactly OSL tokens so runs are comparable.
set -euo pipefail

URL="${URL:-http://localhost:8000}"
MODEL="${MODEL:?set MODEL to the served model name}"
ISL="${ISL:-2048}"
OSL="${OSL:-256}"
CONCURRENCY="${CONCURRENCY:-1 4 16 64 128}"
OUT="${OUT:-runs/aiperf/$(date -u +%Y%m%dT%H%M%SZ)}"

command -v aiperf >/dev/null || { echo "aiperf not found: pip install aiperf" >&2; exit 1; }

for c in ${CONCURRENCY}; do
  requests=$(( c * 10 > 100 ? c * 10 : 100 ))
  echo "== concurrency ${c} (${requests} requests)" >&2
  aiperf profile \
    --model "${MODEL}" \
    --url "${URL}" \
    --endpoint-type chat \
    --streaming \
    --synthetic-input-tokens-mean "${ISL}" \
    --output-tokens-mean "${OSL}" \
    --extra-inputs '{"ignore_eos": true}' \
    --concurrency "${c}" \
    --request-count "${requests}" \
    --warmup-request-count "${c}" \
    --artifact-dir "${OUT}/c${c}"
done

echo "Artifacts in ${OUT}. Each c*/ folder has profile_export_aiperf.json and .csv." >&2
