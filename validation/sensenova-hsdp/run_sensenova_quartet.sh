#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/sensenova-hsdp-validation/u15"
model=/home/kxqandccx/sensenova-7598-20260917/models/SenseNova-U1.5-8B-MoT
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for arm in A0 P0 P1 A1; do
    export CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/dev/shm/0z5a-rope-broadcast
    flags=()
    if [[ $arm == P* ]]; then
        export CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/dev/shm/0z5a-sensenova-e94dafe
        flags=(--hsdp)
    fi
    out="$run_root/$arm"
    mkdir -p "$out"
    export TRITON_CACHE_DIR="$out/triton" TORCHINDUCTOR_CACHE_DIR="$out/inductor" VLLM_CACHE_ROOT="$out/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$out/gpu.csv" &
    monitor_pid=$!
    nvidia-smi pmon -i 2,3 -s um -d 1 > "$out/process-utilization.txt" &
    process_monitor_pid=$!
    trap 'kill "$monitor_pid" "$process_monitor_pid" 2>/dev/null || true' EXIT
    cd "$PYTHONPATH"
    timeout --kill-after=30s 14400 "$python_bin" "$root/sensenova-hsdp-validation/sensenova_benchmark.py" \
        --model "$model" --output "$out" "${flags[@]}" > "$out/run.log" 2>&1
    kill "$monitor_pid" "$process_monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$out/status"
done
printf 'complete\n' > "$run_root/quartet.status"
