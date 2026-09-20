#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
run_root="$root/nextstep-vae-feasibility"
python_bin=/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python
while kill -0 3399484 2>/dev/null; do sleep 30; done
test "$(cat "$root/ring-hsdp-validation/ring-hsdp.status")" = complete
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/dev/shm/0z5a-nextstep-43c4e79:$run_root"
export NEXTSTEP_LATENT_PATH="$run_root/ar-latent.pt"
cd /dev/shm/0z5a-nextstep-43c4e79
timeout --kill-after=30s 14400 "$python_bin" "$run_root/capture_generation.py" \
    --model "$root/models/nextstep" --output "$run_root" > "$run_root/capture.log" 2>&1
timeout --kill-after=30s 3600 "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 \
    "$run_root/validate_real_latent.py" --model "$root/models/nextstep" \
    --latent "$run_root/ar-latent.pt" --output "$run_root/results.json" > "$run_root/validate.log" 2>&1
"$python_bin" "$run_root/summarize.py" --input "$run_root/results.json" --output "$run_root/RESULTS.md"
printf 'complete\n' > "$run_root/nextstep-vae.status"
