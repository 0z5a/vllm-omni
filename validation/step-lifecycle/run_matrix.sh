#!/usr/bin/env bash
set -euo pipefail
run_root=/home/kxqandccx/omni-1217-20260919/step-lifecycle-validation
source_dir=/dev/shm/0z5a-step-lifecycle-bbd391d
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
model=/home/kxqandccx/vllm-omni-7452-7506-20260915/0z5a/quant-three-20260920/helios/model
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$source_dir:$run_root"
cd "$run_root"
sha256sum -c source.sha256
sha256sum -c runtime.sha256
mkdir -p evidence
run_arm() {
    local arm="$1" mode="$2"
    shift 2
    export CUDA_VISIBLE_DEVICES=2
    if [[ $mode == *hsdp ]]; then export CUDA_VISIBLE_DEVICES=2,3; fi
    test ! -e "$run_root/evidence/$arm"
    export TRITON_CACHE_DIR="$run_root/cache/$arm/triton" TORCHINDUCTOR_CACHE_DIR="$run_root/cache/$arm/inductor" VLLM_CACHE_ROOT="$run_root/cache/$arm/vllm"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$run_root/evidence/$arm.gpu.csv" &
    local monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --kill-after=30s 21600 "$python_bin" "$run_root/benchmark.py" --model "$model" --mode "$mode" --output "$run_root/evidence/$arm" "$@" > "$run_root/evidence/$arm.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$run_root/evidence/$arm/status"
}
for mode in native step hsdp step-hsdp module step-module; do run_arm "$mode-probe" "$mode" --probe; done
"$python_bin" verify.py evidence --probes-only
for mode in native step hsdp step-hsdp module step-module; do run_arm "${mode}0" "$mode"; done
for mode in step-module module step-hsdp hsdp step native; do run_arm "${mode}1" "$mode"; done
"$python_bin" verify.py evidence
printf 'complete\n' > evidence/matrix.status
