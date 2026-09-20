#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/hidream-full-admission"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
while kill -0 3700908 2>/dev/null; do sleep 30; done
test "$(cat "$root/nextstep-vae-feasibility/nextstep-vae.status")" = complete
test "$(cat "$root/models/.hidream-downloads.status")" = complete
model="$root/models/hidream-i1-full"
llama="$root/models/llama-3.1-8b-instruct"

run_mode() {
    mode=$1
    source=$2
    devices=$3
    mode_root="$run_root/$mode"
    mkdir -p "$mode_root"
    export CUDA_VISIBLE_DEVICES="$devices"
    export PYTHONPATH="$source:$run_root"
    export TRITON_CACHE_DIR="$mode_root/triton"
    export TORCHINDUCTOR_CACHE_DIR="$mode_root/inductor"
    export VLLM_CACHE_ROOT="$mode_root/vllm-cache"
    nvidia-smi -i "$devices" --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
        --format=csv -l 1 > "$mode_root/gpu.csv" &
    monitor_pid=$!
    if timeout --kill-after=30s 10800 "$python_bin" "$run_root/benchmark.py" \
        --model "$model" --llama "$llama" --mode "$mode" --output "$mode_root" \
        > "$mode_root/run.log" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    kill "$monitor_pid" 2>/dev/null || true
    printf '%s\n' "$exit_code" > "$mode_root/exit-code"
}

run_mode hsdp2 /dev/shm/0z5a-hidream-5410 2,3
run_mode cache /dev/shm/0z5a-hidream-5408 2
run_mode cfg2 /dev/shm/0z5a-hidream-5409 2,3
run_mode vae2 /dev/shm/0z5a-hidream-wiring-patch 2,3
"$python_bin" "$run_root/summarize.py" --root "$run_root" --output "$run_root/RESULTS.md"
printf 'complete\n' > "$run_root/hidream-admission.status"
