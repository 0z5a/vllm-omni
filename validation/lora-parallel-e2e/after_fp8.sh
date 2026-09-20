#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 1147152 -o args= | rg -Fxq "bash $root/fp8-hsdp-validation/after_qwen_cfg.sh"; do sleep 30; done
test "$(cat "$root/fp8-hsdp-validation/evidence/matrix.status")" = complete
cd "$root/lora-parallel-validation"
sha256sum -c source.sha256
bash run_features.sh
