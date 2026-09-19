#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/cachedit-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-compat-698f716:$run_root"
cd /dev/shm/0z5a-compat-698f716
for arm in base0 cache0 hsdp0 both0 both1 hsdp1 cache1 base1 cache-probe both-probe cache-nohit both-nohit; do
    flags=()
    if [[ $arm == hsdp* || $arm == both* ]]; then flags+=(--hsdp); fi
    if [[ $arm == cache* || $arm == both* ]]; then flags+=(--cache-dit); fi
    if [[ $arm == *-probe || $arm == *-nohit ]]; then flags+=(--probe); fi
    if [[ $arm == *-nohit ]]; then flags+=(--no-hit); fi
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton" TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor" VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 5400 "$python_bin" "$run_root/benchmark.py" \
        --model /home/kxqandccx/omni-lora-l20-20260914/model --output "$arm_dir" "${flags[@]}" \
        > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$run_root/evidence/quartet.status"
