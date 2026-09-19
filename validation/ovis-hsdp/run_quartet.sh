#!/usr/bin/env bash
set -euo pipefail

run_root=/home/kxqandccx/omni-1217-20260919
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=4,5
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for arm in A0 P0 P1 A1; do
    source_dir=baseline
    options=()
    if [[ $arm == P* ]]; then
        source_dir=patch
        options=(--hsdp)
    fi
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    export PYTHONPATH="$run_root/$source_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton"
    export TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor"
    export VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 4,5 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    cd "$run_root/$source_dir"
    timeout --kill-after=30s 2700 "$python_bin" "$run_root/benchmark.py" \
        --model "$run_root/model" --output "$arm_dir" "${options[@]}" > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$run_root/evidence/quartet.status"
