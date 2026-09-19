#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 509527 -o args= | grep -Fq "$root/after_current_models.sh"; do sleep 30; done
test "$(cat "$root/glm-validation/evidence/quartet.status")" = complete
bash "$root/cachedit-validation/run_quartet.sh"
