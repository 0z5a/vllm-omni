#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 1068304 -o args= | rg -Fxq "bash $root/qwen-lora-cfg-validation/after_lora_cache.sh"; do sleep 30; done
test "$(cat "$root/qwen-lora-cfg-validation/evidence/quartet.status")" = complete
cd "$root/fp8-hsdp-validation"
sha256sum -c source.sha256
bash run_matrix.sh
