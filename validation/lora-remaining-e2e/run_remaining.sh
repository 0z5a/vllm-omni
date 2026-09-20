#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/lora-remaining-validation"
source_dir=/dev/shm/0z5a-compat-698f716
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
model=/home/kxqandccx/omni-lora-l20-20260914/model
adapter_a="$root/models/lora/helio"
adapter_b="$root/models/lora/fuli"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$source_dir:$run_root:$root/lora-parallel-validation"
cd "$run_root"
sha256sum -c source.sha256
(cd ../lora-parallel-validation && sha256sum -c "$run_root/dependencies.sha256")
mkdir -p evidence

run_arm() {
    local arm="$1" feature="$2"
    shift 2
    export CUDA_VISIBLE_DEVICES=2
    if [[ $feature == u2-* ]]; then export CUDA_VISIBLE_DEVICES=2,3; fi
    test ! -e "$run_root/evidence/$arm"
    export TRITON_CACHE_DIR="$run_root/cache/$arm/triton" TORCHINDUCTOR_CACHE_DIR="$run_root/cache/$arm/inductor" VLLM_CACHE_ROOT="$run_root/cache/$arm/vllm"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$run_root/evidence/$arm.gpu.csv" &
    local monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 14400 "$python_bin" benchmark.py \
        --model "$model" --adapter-a "$adapter_a" --adapter-b "$adapter_b" \
        --feature "$feature" --output "$run_root/evidence/$arm" "$@" \
        > "$run_root/evidence/$arm.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$run_root/evidence/$arm/status"
}

for feature in native layer module fp8 u2-native u2-vae2; do
    run_arm "$feature-probe" "$feature" --probe
    "$python_bin" verify_probe.py "$run_root/evidence/$feature-probe" --feature "$feature"
done

run_arm layer-ref0 native
run_arm layer-new0 layer
run_arm layer-new1 layer
run_arm layer-ref1 native
run_arm module-ref0 native
run_arm module-new0 module
run_arm module-new1 module
run_arm module-ref1 native
run_arm fp8-ref0 native
run_arm fp8-new0 fp8
run_arm fp8-new1 fp8
run_arm fp8-ref1 native
run_arm vae-ref0 u2-native
run_arm vae-new0 u2-vae2
run_arm vae-new1 u2-vae2
run_arm vae-ref1 u2-native
"$python_bin" summarize.py evidence
printf 'complete\n' > evidence/remaining-cases.status
