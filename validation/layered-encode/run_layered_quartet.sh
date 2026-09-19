#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/layered-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/dev/shm/0z5a-layered-58dfc80
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 "$run_root/layered_cuda_parity.py" > "$run_root/cuda-parity.log" 2>&1
for arm in A0 P0 P1 A1; do
    sha=698f716
    export PYTHONPATH=/dev/shm/0z5a-compat-698f716
    flags=()
    if [[ $arm == P* ]]; then
        sha=58dfc80
        export PYTHONPATH=/dev/shm/0z5a-layered-58dfc80
        flags=(--parallel-vae)
    fi
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton" TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor" VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    nvidia-smi pmon -i 2,3 -s um -d 1 > "$arm_dir/process-utilization.txt" &
    process_monitor_pid=$!
    trap 'kill "$monitor_pid" "$process_monitor_pid" 2>/dev/null || true' EXIT
    cd "$PYTHONPATH"
    timeout --kill-after=30s 14400 "$python_bin" "$run_root/layered_benchmark.py" \
        --model /home/kxqandccx/omni-1217-20260919/models/qwen-image-layered \
        --source-sha "$sha" --output "$arm_dir" "${flags[@]}" > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid" "$process_monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$run_root/evidence/quartet.status"
