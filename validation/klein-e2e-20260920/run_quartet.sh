#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/klein-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
model=/home/kxqandccx/omni-1217-20260919/models/flux2-klein-4b
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/dev/shm/0z5a-klein-7464
cd "$PYTHONPATH"
mkdir -p "$run_root/components"
timeout --kill-after=30s 7200 "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 \
    "$run_root/vae_contract.py" --model "$model" --output "$run_root/components" > "$run_root/components/run.log" 2>&1
printf 'complete\n' > "$run_root/components/status"
for arm in A0 P0 P1 A1; do
    flags=()
    export PYTHONPATH=/dev/shm/0z5a-compat-698f716
    if [[ $arm == P* ]]; then
        export PYTHONPATH=/dev/shm/0z5a-klein-7464
        flags+=(--parallel-vae)
    fi
    cd "$PYTHONPATH"
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton" TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor" VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 7200 "$python_bin" "$run_root/benchmark.py" --model "$model" --output "$arm_dir" "${flags[@]}" > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$run_root/evidence/quartet.status"
