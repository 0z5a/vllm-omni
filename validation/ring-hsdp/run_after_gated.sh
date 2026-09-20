#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/ring-hsdp-validation"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
while ps -p 3074069 -o args= | rg -q 'run_after_lora\.sh$'; do sleep 30; done
test "$(cat "$root/gated-model-validation/evidence/gated-models.status")" = complete
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-compat-698f716:$run_root"
cd /dev/shm/0z5a-compat-698f716
mkdir -p "$run_root/evidence"
run_arm() {
    local arm="$1"
    shift
    local output="$run_root/evidence/$arm"
    test ! -e "$output"
    export TRITON_CACHE_DIR="$output/triton" TORCHINDUCTOR_CACHE_DIR="$output/inductor" VLLM_CACHE_ROOT="$output/vllm"
    timeout --kill-after=30s 14400 "$python_bin" "$run_root/benchmark.py" \
        --model /home/kxqandccx/omni-lora-l20-20260914/model --output "$output" "$@" \
        > "$run_root/evidence/$arm.log" 2>&1
    printf 'complete\n' > "$output/status"
}
run_arm native-probe --probe
run_arm hsdp-probe --probe --hsdp
"$python_bin" "$run_root/verify.py" "$run_root/evidence/native-probe" "$run_root/evidence/hsdp-probe"
run_arm native0
run_arm hsdp0 --hsdp
run_arm hsdp1 --hsdp
run_arm native1
printf 'complete\n' > "$run_root/evidence/ring-hsdp.status"
