#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 1432273 -o args= | rg -Fxq "bash $root/lora-parallel-validation/after_fp8.sh"; do sleep 30; done
test "$(cat "$root/lora-parallel-validation/evidence/parallel-cases.status")" = complete
cd "$root/step-lifecycle-validation"
bash run_matrix.sh
