#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/lora-parallel-validation"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-compat-698f716:$run_root"
cd /dev/shm/0z5a-compat-698f716
mkdir -p "$run_root/evidence"
common=(--model /home/kxqandccx/omni-lora-l20-20260914/model --adapter-a "$root/models/lora/helio" --adapter-b "$root/models/lora/fuli")
run_arm() {
    local arm="$1" feature="$2"
    shift 2
    local arm_dir="$run_root/evidence/$arm"
    test ! -e "$arm_dir"
    export CUDA_VISIBLE_DEVICES=2,3
    if [[ $feature == none ]]; then export CUDA_VISIBLE_DEVICES=2; fi
    export TRITON_CACHE_DIR="$run_root/cache/$arm/triton" TORCHINDUCTOR_CACHE_DIR="$run_root/cache/$arm/inductor" VLLM_CACHE_ROOT="$run_root/cache/$arm/vllm"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$run_root/evidence/$arm.gpu.csv" &
    local monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 14400 "$python_bin" "$run_root/benchmark.py" "${common[@]}" --feature "$feature" --output "$arm_dir" "$@" > "$run_root/evidence/$arm.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
}
run_arm native-probe none --probe
"$python_bin" "$run_root/verify_switch.py" "$run_root/evidence/native-probe" --ranks 1
for feature in ulysses ring tp hsdp; do
    run_arm "$feature-probe" "$feature" --probe
    "$python_bin" "$run_root/verify_parallel.py" --native "$run_root/evidence/native-probe" --parallel "$run_root/evidence/$feature-probe"
    run_arm "$feature-A0" none
    run_arm "$feature-P0" "$feature"
    run_arm "$feature-P1" "$feature"
    run_arm "$feature-A1" none
    printf 'complete\n' > "$run_root/evidence/$feature.status"
done
printf 'complete\n' > "$run_root/evidence/parallel-cases.status"
