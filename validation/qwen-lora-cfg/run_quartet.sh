#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/qwen-lora-cfg-validation"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-qwen-lora-3ea5152:$run_root"
cd /dev/shm/0z5a-qwen-lora-3ea5152
mkdir -p "$run_root/evidence"
common=(--model "$root/models/qwen-image-edit-2509" --adapter-a "$root/models/lora/edit-r1" --adapter-b "$root/models/lora/cocoedit")
run_arm() {
    local arm="$1" degree="$2"
    shift 2
    local arm_dir="$run_root/evidence/$arm"
    test ! -e "$arm_dir"
    export CUDA_VISIBLE_DEVICES=2
    if [[ $degree == 2 ]]; then export CUDA_VISIBLE_DEVICES=2,3; fi
    export TRITON_CACHE_DIR="$run_root/cache/$arm/triton" TORCHINDUCTOR_CACHE_DIR="$run_root/cache/$arm/inductor" VLLM_CACHE_ROOT="$run_root/cache/$arm/vllm"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$run_root/evidence/$arm.gpu.csv" &
    local monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 86400 "$python_bin" "$run_root/benchmark.py" "${common[@]}" --cfg-degree "$degree" --output "$arm_dir" "$@" > "$run_root/evidence/$arm.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
}
run_arm native-probe 1 --probe
"$python_bin" "$run_root/verify_switch.py" "$run_root/evidence/native-probe" --ranks 1
run_arm cfg-probe 2 --probe
"$python_bin" "$run_root/verify_cfg.py" --native "$run_root/evidence/native-probe" --parallel "$run_root/evidence/cfg-probe"
run_arm A0 1
run_arm P0 2
run_arm P1 2
run_arm A1 1
printf 'complete\n' > "$run_root/evidence/quartet.status"
