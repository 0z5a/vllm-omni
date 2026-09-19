#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/omnigen2-reference-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for arm in A0 P0 P1 A1; do
    sha=15840a6
    if [[ $arm == P* ]]; then sha=f2c1684; fi
    source_dir="/dev/shm/0z5a-omnigen2-$sha"
    export PYTHONPATH="$run_root:$source_dir"
    arm_dir="$run_root/evidence/$arm"
    mkdir -p "$arm_dir"
    export TRITON_CACHE_DIR="$arm_dir/triton" TORCHINDUCTOR_CACHE_DIR="$arm_dir/inductor" VLLM_CACHE_ROOT="$arm_dir/vllm-cache"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$arm_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    cd "$source_dir"
    timeout --kill-after=30s 5400 "$python_bin" "$run_root/benchmark.py" --model /home/kxqandccx/omnigen2-encoder-fp8-20260915/model --source-sha "$sha" --output "$arm_dir" > "$arm_dir/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
printf 'complete\n' > "$run_root/evidence/quartet.status"
