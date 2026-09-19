#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 363061 -o args= | grep -Fq "$root/klein-validation/run_quartet.sh"; do sleep 30; done
test "$(cat "$root/klein-validation/evidence/quartet.status")" = complete
bash "$root/nextstep-validation/run_quartet.sh"
bash "$root/sensenova-hsdp-validation/run_sensenova_quartet.sh"
bash "$root/layered-validation/run_layered_quartet.sh"
bash "$root/qwen-edit-validation/run_qwen_edit_quartets.sh"
bash "$root/compat-validation/run_compat_quartet.sh"
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=/dev/shm/0z5a-klein-7464
cd "$PYTHONPATH"
mkdir -p "$root/ernie/vae-contract"
timeout --kill-after=30s 7200 /home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
    "$root/klein-validation/vae_contract.py" --model "$root/ernie/model" --output "$root/ernie/vae-contract" > "$root/ernie/vae-contract/run.log" 2>&1
printf 'complete\n' > "$root/ernie/vae-contract/status"
