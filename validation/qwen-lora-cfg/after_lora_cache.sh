#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 854106 -o args= | rg -Fxq 'bash after_current.sh'; do sleep 30; done
test "$(cat "$root/lora-switch-validation/evidence/cache-cases.status")" = complete
cd "$root/qwen-lora-cfg-validation"
sha256sum -c source.sha256
bash run_quartet.sh
