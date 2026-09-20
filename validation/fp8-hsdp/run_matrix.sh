#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/fp8-hsdp-validation
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-compat-698f716:$run_root"
cd /dev/shm/0z5a-compat-698f716
mkdir -p "$run_root/evidence"
for arm in base-probe fp8-probe hsdp-probe both-probe base0 fp80 hsdp0 both0 both1 hsdp1 fp81 base1; do
    flags=()
    if [[ $arm == hsdp* || $arm == both* ]]; then flags+=(--hsdp); fi
    if [[ $arm == fp8* || $arm == both* ]]; then flags+=(--fp8); fi
    if [[ $arm == *-probe ]]; then flags+=(--probe); fi
    arm_dir="$run_root/evidence/$arm"
    test ! -e "$arm_dir"
    export TRITON_CACHE_DIR="$run_root/cache/$arm/triton" TORCHINDUCTOR_CACHE_DIR="$run_root/cache/$arm/inductor" VLLM_CACHE_ROOT="$run_root/cache/$arm/vllm"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$run_root/evidence/$arm.gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 14400 "$python_bin" "$run_root/benchmark.py" \
        --model /home/kxqandccx/omni-lora-l20-20260914/model --output "$arm_dir" "${flags[@]}" \
        > "$run_root/evidence/$arm.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$arm_dir/status"
done
"$python_bin" "$run_root/summarize.py" "$run_root/evidence"
printf 'complete\n' > "$run_root/evidence/matrix.status"
