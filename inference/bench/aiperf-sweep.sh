#!/usr/bin/env bash
# Sweep request concurrency against an OpenAI-compatible endpoint (vLLM, SGLang,
# TensorRT-LLM or the Dynamo frontend) with AIPerf, and keep every run's artifacts.
#
#   kubectl port-forward svc/qwen3-disagg-frontend 8000:8000 &
#   MODEL=Qwen/Qwen3-32B inference/bench/aiperf-sweep.sh
#
# `bcp rental run` calls this for every inference config. Defaults model a chat/RAG-style
# load: 2k tokens in, 256 out. ignore_eos makes every request produce exactly OSL tokens, and
# a fixed seed and tokenizer give every server the same prompts, so runs are comparable.
set -euo pipefail

URL="${URL:-http://localhost:8000}"
MODEL="${MODEL:?set MODEL to the served model name}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
ISL="${ISL:-2048}"
OSL="${OSL:-256}"
CONCURRENCY="${CONCURRENCY:-1 4 16 64 128}"
REQUESTS_PER_CONCURRENCY="${REQUESTS_PER_CONCURRENCY:-8}"
MIN_REQUESTS="${MIN_REQUESTS:-32}"
RANDOM_SEED="${RANDOM_SEED:-42}"
OUT="${OUT:-runs/aiperf/$(date -u +%Y%m%dT%H%M%SZ)}"

command -v aiperf >/dev/null || { echo "aiperf not found: pip install aiperf" >&2; exit 1; }

for c in ${CONCURRENCY}; do
  requests=$(( c * REQUESTS_PER_CONCURRENCY > MIN_REQUESTS ? c * REQUESTS_PER_CONCURRENCY : MIN_REQUESTS ))
  echo "== concurrency ${c} (${requests} requests)" >&2
  aiperf profile \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --url "${URL}" \
    --endpoint-type chat \
    --streaming \
    --synthetic-input-tokens-mean "${ISL}" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "${OSL}" \
    --output-tokens-stddev 0 \
    --extra-inputs '{"ignore_eos": true}' \
    --concurrency "${c}" \
    --request-count "${requests}" \
    --warmup-request-count "${c}" \
    --random-seed "${RANDOM_SEED}" \
    --ui-type none \
    --artifact-dir "${OUT}/c${c}"
done

echo "Artifacts in ${OUT}. Each c*/ folder has profile_export_aiperf.json and .csv." >&2
