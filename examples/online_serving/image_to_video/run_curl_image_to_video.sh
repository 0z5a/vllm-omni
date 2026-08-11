#!/bin/bash
# Shared image-to-video curl example using the async video job API.

set -euo pipefail

INPUT_IMAGE="${INPUT_IMAGE:-../../offline_inference/image_to_video/qwen-bear.png}"
BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-image_to_video_output.mp4}"
PROMPT="${PROMPT:-A bear playing with yarn, smooth motion}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
VIDEO_SECONDS="${VIDEO_SECONDS:-}"
SIZE="${SIZE:-}"
FPS="${FPS:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-}"
FLOW_SHIFT="${FLOW_SHIFT:-}"
SEED="${SEED:-}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    exit 1
fi

create_cmd=(
  curl -sS -X POST "${BASE_URL}/v1/videos"
  -H "Accept: application/json"
  -F "prompt=${PROMPT}"
  -F "input_reference=@${INPUT_IMAGE}"
)

append_optional_field() {
  local name="$1"
  local value="$2"
  if [ -n "${value}" ]; then
    create_cmd+=(-F "${name}=${value}")
  fi
}

append_optional_field "seconds" "${VIDEO_SECONDS}"
append_optional_field "size" "${SIZE}"
append_optional_field "fps" "${FPS}"
append_optional_field "num_inference_steps" "${NUM_INFERENCE_STEPS}"
append_optional_field "guidance_scale" "${GUIDANCE_SCALE}"
append_optional_field "guidance_scale_2" "${GUIDANCE_SCALE_2}"
append_optional_field "boundary_ratio" "${BOUNDARY_RATIO}"
append_optional_field "flow_shift" "${FLOW_SHIFT}"
append_optional_field "seed" "${SEED}"

if [ -n "${NEGATIVE_PROMPT}" ]; then
  create_cmd+=(-F "negative_prompt=${NEGATIVE_PROMPT}")
fi

if [ -n "${SAMPLE_SOLVER}" ]; then
  create_cmd+=(-F "extra_params={\"sample_solver\":\"${SAMPLE_SOLVER}\"}")
fi

create_response="$("${create_cmd[@]}")"
video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"
