#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/pending"
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 "$root/vae_parallel_parity.py" > "$root/vae-validation/components.log" 2>&1
for model in ernie_image omnigen2; do
    driver="$root/ernie/benchmark_rebased.py"
    weights="$root/ernie/model"
    if [[ $model == omnigen2 ]]; then
        driver="$root/benchmark_omnigen2_vae.py"
        weights=/home/kxqandccx/omnigen2-encoder-fp8-20260915/model
    fi
    for arm in A0 P0 P1 A1; do
        export PYTHONPATH="$root/rebased/native"
        options=()
        if [[ $arm == P* ]]; then
            export PYTHONPATH="$root/vae-validation/$model/source"
            options=(--parallel-vae)
        fi
        out="$root/vae-validation/$model/evidence/$arm"
        mkdir -p "$out"
        export TRITON_CACHE_DIR="$out/triton" TORCHINDUCTOR_CACHE_DIR="$out/inductor" VLLM_CACHE_ROOT="$out/vllm-cache"
        nvidia-smi -i 2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$out/gpu.csv" &
        monitor_pid=$!
        trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
        cd "$PYTHONPATH"
        timeout --kill-after=30s 2700 "$python_bin" "$driver" --model "$weights" --output "$out" "${options[@]}" > "$out/run.log" 2>&1
        kill "$monitor_pid"
        trap - EXIT
        printf "complete\n" > "$out/status"
    done
done
