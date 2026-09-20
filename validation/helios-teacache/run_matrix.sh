#!/usr/bin/env bash
set -euo pipefail
root=/workspace/omni-1217-20260920/evidence/helios-teacache
source_dir=/workspace/omni-1217-20260920/worktrees/vllm-omni
python_bin=/workspace/omni-1217-20260920/venv/bin/python
model=/workspace/omni-1217-20260920/models/helios-distilled
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONPATH="$source_dir:$root"
mkdir -p "$root/evidence" "$root/cache"
cd "$root"

run_arm() {
    local arm="$1" mode="$2" threshold="$3" probe="$4"
    local target="$root/evidence/$arm"
    test ! -e "$target"
    export TRITON_CACHE_DIR="$root/cache/$arm/triton"
    export TORCHINDUCTOR_CACHE_DIR="$root/cache/$arm/inductor"
    export VLLM_CACHE_ROOT="$root/cache/$arm/vllm"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
    nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$target.gpu.csv" &
    local monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    local args=(--model "$model" --mode "$mode" --output "$target")
    if [[ -n "$threshold" ]]; then args+=(--threshold "$threshold"); fi
    if [[ "$probe" == 1 ]]; then args+=(--probe); fi
    timeout --kill-after=30s 43200 "$python_bin" benchmark.py "${args[@]}" > "$target.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$target/status"
}

run_arm baseline-probe baseline '' 1
run_arm nohit-probe nohit 0.2 1
for threshold in 0.05 0.10 0.20 0.30; do
    run_arm "cache-${threshold}-probe" cache "$threshold" 1
done
run_arm baseline-0 baseline '' 0
for threshold in 0.05 0.10 0.20 0.30; do
    run_arm "cache-${threshold}-0" cache "$threshold" 0
done
for threshold in 0.30 0.20 0.10 0.05; do
    run_arm "cache-${threshold}-1" cache "$threshold" 0
done
run_arm baseline-1 baseline '' 0
"$python_bin" verify.py "$root/evidence" | tee "$root/results.md"
printf 'complete\n' > "$root/evidence/matrix.status"
