#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/gated-model-validation"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
while ps -p 2477741 -o args= | rg -q 'after_step_lifecycle\.sh$'; do sleep 30; done
test "$(cat "$root/lora-remaining-validation/evidence/remaining-cases.status")" = complete
test "$(cat "$root/models/.gated-downloads.status")" = complete
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

run_arm() {
    local scope="$1" arm="$2" source="$3" source_sha="$4" model="$5" pipeline="$6" revision="$7" feature="$8"
    local output="$run_root/evidence/$scope-$arm"
    test ! -e "$output"
    mkdir -p "$output"
    export PYTHONPATH="$source"
    export TRITON_CACHE_DIR="$output/triton" TORCHINDUCTOR_CACHE_DIR="$output/inductor" VLLM_CACHE_ROOT="$output/vllm"
    nvidia-smi -i 2,3 --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$output/gpu.csv" &
    local monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    local feature_arg=()
    if [[ $feature == yes ]]; then feature_arg=(--feature); fi
    timeout --kill-after=30s 28800 "$python_bin" "$run_root/benchmark.py" \
        --model "$model" --pipeline "$pipeline" --source-sha "$source_sha" \
        --checkpoint-revision "$revision" --output "$output" "${feature_arg[@]}" \
        > "$output/run.log" 2>&1
    kill "$monitor_pid"
    trap - EXIT
    printf 'complete\n' > "$output/status"
}

base=/dev/shm/0z5a-compat-698f716
flux=/dev/shm/0z5a-flux1-9054999
sd3=/dev/shm/0z5a-sd35-892df59
for spec in \
    "A2S flux1-schnell flux 741f7c3ce8b383c54771c7003378a50191e9efe9" \
    "A2D flux1-dev flux 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21" \
    "A45 flux1-kontext kontext 24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"; do
    read -r scope slug pipeline revision <<< "$spec"
    run_arm "$scope" ref0 "$base" 698f7160125d3071b8c0eef69b1b03fa8dfba766 "$root/models/$slug" "$pipeline" "$revision" no
    run_arm "$scope" new0 "$flux" 9054999b62866635246c88204cb514863134a242 "$root/models/$slug" "$pipeline" "$revision" yes
    run_arm "$scope" new1 "$flux" 9054999b62866635246c88204cb514863134a242 "$root/models/$slug" "$pipeline" "$revision" yes
    run_arm "$scope" ref1 "$base" 698f7160125d3071b8c0eef69b1b03fa8dfba766 "$root/models/$slug" "$pipeline" "$revision" no
done
run_arm H1 ref0 "$base" 698f7160125d3071b8c0eef69b1b03fa8dfba766 "$root/models/sd35-medium" sd3 b940f670f0eda2d07fbb75229e779da1ad11eb80 no
run_arm H1 new0 "$sd3" 892df593a09395d2852c9c98433c59df769caaa1 "$root/models/sd35-medium" sd3 b940f670f0eda2d07fbb75229e779da1ad11eb80 yes
run_arm H1 new1 "$sd3" 892df593a09395d2852c9c98433c59df769caaa1 "$root/models/sd35-medium" sd3 b940f670f0eda2d07fbb75229e779da1ad11eb80 yes
run_arm H1 ref1 "$base" 698f7160125d3071b8c0eef69b1b03fa8dfba766 "$root/models/sd35-medium" sd3 b940f670f0eda2d07fbb75229e779da1ad11eb80 no
"$python_bin" "$run_root/summarize.py" "$run_root/evidence"
printf 'complete\n' > "$run_root/evidence/gated-models.status"
du -sb "$root/models/flux1-schnell" "$root/models/flux1-dev" \
    "$root/models/flux1-kontext" "$root/models/sd35-medium" > "$run_root/evidence/model-cleanup.tsv"
rm -rf "$root/models/flux1-schnell" "$root/models/flux1-dev" \
    "$root/models/flux1-kontext" "$root/models/sd35-medium"
printf 'complete\n' > "$run_root/evidence/model-cleanup.status"
