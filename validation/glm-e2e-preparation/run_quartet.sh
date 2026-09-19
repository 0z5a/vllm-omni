#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919/glm-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for arm in A0 P0 P1 A1 A-probe P-probe; do
    flags=()
    source_dir=/dev/shm/0z5a-compat-698f716
    config=baseline
    if [[ $arm == P* ]]; then
        source_dir=/dev/shm/0z5a-glm-635df16
        config=parallel
        flags+=(--parallel-vae)
    fi
    if [[ $arm == *probe ]]; then
        config+=-probe
        flags+=(--probe)
    fi
    export PYTHONPATH="$source_dir:$root"
    cd "$source_dir"
    arm_dir="$root/evidence/$arm"
    mkdir -p "$arm_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton" TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor" VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 14400 "$python_bin" "$root/benchmark.py" \
        --model /home/kxqandccx/omni-1217-20260919/models/glm-image \
        --deploy-config "$root/$config.yaml" --output "$arm_dir" "${flags[@]}" > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$root/evidence/quartet.status"
