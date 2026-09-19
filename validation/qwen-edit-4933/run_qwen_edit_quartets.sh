#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/qwen-edit-validation"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/dev/shm/0z5a-qwen-edit-4933
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 "$run_root/qwen_edit_cuda_parity.py" > "$run_root/cuda-parity.log" 2>&1
for model in qwen-image-edit qwen-image-edit-2509; do
    for arm in A0 P0 P1 A1 A-probe P-probe; do
        flags=()
        if [[ $arm == *-probe ]]; then flags+=(--probe); fi
        export PYTHONPATH="/dev/shm/0z5a-compat-698f716:$run_root"
        if [[ $arm == P* ]]; then
            export PYTHONPATH="/dev/shm/0z5a-qwen-edit-4933:$run_root"
            flags+=(--parallel-vae)
        fi
        if [[ $model == qwen-image-edit-2509 ]]; then flags+=(--plus); fi
        out="$run_root/$model/$arm"
        mkdir -p "$out"
        export TRITON_CACHE_DIR="$out/triton" TORCHINDUCTOR_CACHE_DIR="$out/inductor" VLLM_CACHE_ROOT="$out/vllm-cache"
        nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$out/gpu.csv" &
        monitor_pid=$!
        nvidia-smi pmon -i 2,3 -s um -d 1 > "$out/process-utilization.txt" &
        process_monitor_pid=$!
        trap 'kill "$monitor_pid" "$process_monitor_pid" 2>/dev/null || true' EXIT
        cd "${PYTHONPATH%%:*}"
        timeout --kill-after=30s 14400 "$python_bin" "$run_root/qwen_edit_benchmark.py" \
            --model "$root/models/$model" --output "$out" "${flags[@]}" > "$out/run.log" 2>&1
        kill "$monitor_pid" "$process_monitor_pid"
        trap - EXIT
        printf 'complete\n' > "$out/status"
    done
    printf 'complete\n' > "$run_root/$model/quartet.status"
done
