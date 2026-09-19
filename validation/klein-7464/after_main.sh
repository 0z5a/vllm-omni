#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 4158704 -o args= | grep -Fq "$root/run_remaining_20260920.sh"; do sleep 30; done
test "$(cat "$root/compat-validation/evidence/quartet.status")" = complete
bash "$root/klein-validation/run_quartet.sh"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export PYTHONPATH=/dev/shm/0z5a-klein-7464
cd "$PYTHONPATH"
mkdir -p "$root/ernie/vae-contract"
timeout --kill-after=30s 7200 /home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
    "$root/klein-validation/vae_contract.py" --model "$root/ernie/model" --output "$root/ernie/vae-contract" > "$root/ernie/vae-contract/run.log" 2>&1
printf 'complete\n' > "$root/ernie/vae-contract/status"
