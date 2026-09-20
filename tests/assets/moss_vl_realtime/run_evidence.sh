#!/bin/bash
# Produce the full MOSS-VL-Realtime evidence set on one RTX 5090 host.
#
# In order: checkpoint hash verification, the reference offline capture, the
# native replay with comparison, the stage parity probes, the paced realtime
# capture, the native paced replay, the timeline comparison, and the baseline
# aggregation. The device comes from MOSS_ALLOWED_GPUS (default "1 2 3", i.e.
# never GPU 0 or 4-7 on a shared host) and the freest of those is used; pass
# MOSS_GPU to pin one. Nothing here installs packages or touches other GPUs.
#
# Usage:
#   MOSS_REPO=~/vllm-omni MOSS_CKPT=~/ckpt/MOSS-VL-Realtime \
#   MOSS_REF_SITE=~/refsite MOSS_PYTHON=~/0z5a/bin/python \
#     bash tests/assets/moss_vl_realtime/run_evidence.sh [output-root]
set -u
REPO=${MOSS_REPO:-$PWD}
CKPT=${MOSS_CKPT:?set MOSS_CKPT to the checkpoint directory}
REF_SITE=${MOSS_REF_SITE:?set MOSS_REF_SITE to the transformers-4.57 overlay directory}
PY=${MOSS_PYTHON:-python}
ART=${MOSS_ARTIFACTS:-artifacts/moss}
LOG=${MOSS_LOGS:-$ART/logs}
REV=${MOSS_REVISION:-2cb8df5c2adae6b59653bbdd783dc580cf440175}
OUT=${1:-$ART/2cb8df5c-0e016ef7f}
mkdir -p "$LOG" "$OUT"
# The device is chosen from an explicit allow-list, never from "whatever is
# free": on a shared host the free devices may belong to someone else's work.
ALLOWED=${MOSS_ALLOWED_GPUS:-"1 2 3"}
if [ -n "${MOSS_GPU:-}" ]; then
  GPU=$MOSS_GPU
else
  GPU=$(for index in $ALLOWED; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$index"
  done | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
echo "using GPU $GPU"

declare -A WANT=(
  [model-00001-of-00005.safetensors]=3fe2d46a92e3c036e8dfbb2519f65415f8f1ab4f731ffdcb9c8bbff4e66fc3d6
  [model-00002-of-00005.safetensors]=18f3579907933bad721053b3ac405c51045db4539fedcc7ee3c8db79e79f70e5
  [model-00003-of-00005.safetensors]=cecf1e02afe5c7250bb43d6b6e99c7cb430a02afc07a2d697ab1dc4b5f784567
  [model-00004-of-00005.safetensors]=57d794a266a1283d209d054114d9888a6a6b36a0af87fbf314c92b76c89dc294
  [model-00005-of-00005.safetensors]=34009172ba732eb447caf1293126bba35eb934a622677ac7c24bb2d6af462b0d
)

cd "$REPO" || exit 1
echo "=== verify checkpoint $(date +%H:%M:%S)"
for file in "${!WANT[@]}"; do
  got=$(sha256sum "$CKPT/$file" | cut -d' ' -f1)
  if [ "$got" != "${WANT[$file]}" ]; then
    echo "CHECKPOINT_MISMATCH $file"
    exit 1
  fi
done
echo "checkpoint verified"

echo "=== reference capture (offline, sdpa) $(date +%H:%M:%S)"
rm -rf "$OUT"
PYTHONPATH=$REF_SITE CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" \
  tests/assets/moss_vl_realtime/capture_reference.py --checkpoint "$CKPT" --revision "$REV" --out "$OUT" \
  --case prompt_only --case image --case short_video --attn-impl sdpa >"$LOG/e2e-capture.log" 2>&1 \
  || { echo CAPTURE_FAILED; exit 1; }

for case in prompt_only image short_video; do
  run=$(ls -d "$OUT/$case"/*/ | head -1)
  echo "=== native $case $(date +%H:%M:%S)"
  PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPU "$PY" -m vllm_omni.model_executor.models.moss_vl_realtime.runner \
    --checkpoint "$CKPT" --run-dir "$run" --out "$OUT/native/$case" --compare --max-new-tokens 64 \
    >"$LOG/e2e-native-$case.log" 2>&1 || echo "NATIVE_FAILED $case"
  PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPU "$PY" -m vllm_omni.model_executor.models.moss_vl_realtime.parity \
    --checkpoint "$CKPT" --run-dir "$run" --report "$OUT/native/$case/parity-stages.json" \
    >"$LOG/e2e-parity-$case.log" 2>&1 || echo "PARITY_FAILED $case"
done

echo "=== paced realtime capture $(date +%H:%M:%S)"
PYTHONPATH=$REF_SITE CUDA_VISIBLE_DEVICES=$GPU "$PY" tests/assets/moss_vl_realtime/capture_realtime.py \
  --checkpoint "$CKPT" --revision "$REV" --case timestamped_stream --out "$OUT" --playback-speed 1.0 \
  >"$LOG/e2e-realtime.log" 2>&1 || echo "REALTIME_FAILED"

echo "=== native paced replay $(date +%H:%M:%S)"
PYTHONPATH=$REF_SITE:$REPO CUDA_VISIBLE_DEVICES=$GPU "$PY" tests/assets/moss_vl_realtime/replay_paced.py \
  --checkpoint "$CKPT" --prompt "Describe important changes." \
  --frame red.ppm --frame blue.ppm --frame red.ppm --frame blue.ppm \
  --batch-plan tests/assets/moss_vl_realtime/reference/realtime/batch-plan.json \
  --out "$OUT/native-paced" >"$LOG/e2e-native-paced.log" 2>&1 || echo "PACED_FAILED"

echo "=== reference step parity $(date +%H:%M:%S)"
PYTHONPATH=$REF_SITE:$REPO CUDA_VISIBLE_DEVICES=$GPU "$PY" tests/assets/moss_vl_realtime/replay_paced.py \
  --checkpoint "$CKPT" \
  --frame red.ppm --frame blue.ppm --frame red.ppm --frame blue.ppm \
  --reference-steps tests/assets/moss_vl_realtime/reference/realtime/steps.jsonl \
  --out "$OUT/step-parity" >"$LOG/e2e-step-parity.log" 2>&1 || echo "STEP_PARITY_FAILED"

echo "=== paced timeline comparison $(date +%H:%M:%S)"
REF_TRACE=$(ls -d "$OUT/timestamped_stream"/*/ 2>/dev/null | head -1)
NATIVE_TRACE=$(ls -d "$OUT/native-paced"/*/ 2>/dev/null | head -1)
if [ -n "$REF_TRACE" ] && [ -n "$NATIVE_TRACE" ]; then
  PYTHONPATH=$REPO "$PY" -m vllm_omni.model_executor.models.moss_vl_realtime.timeline \
    "$REF_TRACE/events.jsonl" --compare "$NATIVE_TRACE/events.jsonl" \
    --out "$OUT/timeline-comparison.json" >"$LOG/e2e-timeline.log" 2>&1 || echo "TIMELINE_FAILED"
fi

echo "=== baseline aggregation $(date +%H:%M:%S)"
PYTHONPATH=$REPO "$PY" -m vllm_omni.model_executor.models.moss_vl_realtime.baseline \
  --artifacts "$OUT" --native-root "$OUT/native" \
  --realtime-root "$OUT" --realtime-native-root "$OUT/native-paced" \
  --out "$OUT/baseline.md" >"$LOG/e2e-baseline.log" 2>&1 \
  || echo "BASELINE_FAILED"

echo "=== done $(date +%H:%M:%S)"
