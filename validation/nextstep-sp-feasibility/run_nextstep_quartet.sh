#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/nextstep-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-nextstep-9545f40:$run_root"
for arm in TP0 H0 H1 TP1; do
    mode=tp2
    if [[ $arm == H* ]]; then mode=hsdp2; fi
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton" TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor" VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    cd /dev/shm/0z5a-nextstep-9545f40
    timeout --kill-after=30s 14400 "$python_bin" "$run_root/benchmark.py" --model /home/kxqandccx/omni-1217-20260919/models/nextstep --source-sha 9545f40 --mode "$mode" --output "$arm_dir" > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
    if [[ $arm == TP0 ]]; then
        profile_dir="$run_root/prefill-profile"
        mkdir -p "$profile_dir"
        export TRITON_CACHE_DIR="$profile_dir/triton" TORCHINDUCTOR_CACHE_DIR="$profile_dir/inductor" VLLM_CACHE_ROOT="$profile_dir/vllm-cache"
        timeout --kill-after=30s 3600 "$python_bin" "$run_root/benchmark.py" \
            --model /home/kxqandccx/omni-1217-20260919/models/nextstep --source-sha 9545f40 \
            --mode tp2 --profile --smoke --warmups 1 --repeats 1 --output "$profile_dir" > "$profile_dir/run.log" 2>&1
        printf 'complete\n' > "$profile_dir/status"
    fi
done
printf 'complete\n' > "$run_root/evidence/quartet.status"
