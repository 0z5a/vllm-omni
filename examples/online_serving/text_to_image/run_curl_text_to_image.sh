#!/bin/bash
# Shared text-to-image curl example

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
PROMPT="${PROMPT:-a dragon laying over the spine of the Green Mountains of Vermont}"
SIZE="${SIZE:-}"
SEED="${SEED:-}"
OUTPUT_PATH="${OUTPUT_PATH:-text_to_image_output.png}"

curl -sS -X POST "${BASE_URL}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg prompt "${PROMPT}" --arg size "${SIZE}" --arg seed "${SEED}" \
    '{prompt: $prompt}
     + (if $size == "" then {} else {size: $size} end)
     + (if $seed == "" then {} else {seed: ($seed | tonumber)} end)')" \
  | jq -r '.data[0].b64_json' \
  | base64 -d > "${OUTPUT_PATH}"

echo "Saved image to ${OUTPUT_PATH}"
