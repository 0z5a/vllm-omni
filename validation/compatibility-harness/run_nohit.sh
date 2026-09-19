#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 575508 -o args= | grep -Fq "$root/cachedit-validation/after_glm.sh"; do sleep 30; done
test "$(cat "$root/cachedit-validation/evidence/quartet.status")" = complete
run_root="$root/compat-validation"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-compat-698f716:$run_root"
cd /dev/shm/0z5a-compat-698f716
for arm in cache-nohit both-nohit; do
    flags=()
    if [[ $arm == both* ]]; then flags+=(--hsdp); fi
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    timeout --kill-after=30s 5400 /home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python \
        "$run_root/compat_nohit_benchmark.py" --model /home/kxqandccx/omni-lora-l20-20260914/model \
        --output "$arm_dir" --tea-cache --probe "${flags[@]}" > "$arm_dir/run.log" 2>&1
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$run_root/evidence/nohit.status"
