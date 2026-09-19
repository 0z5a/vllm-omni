#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 395864 -o args= | grep -Fq "$root/resume_after_klein.sh"; do sleep 30; done
test "$(cat "$root/ernie/vae-contract/status")" = complete
bash "$root/helios-validation/run_quartet.sh"
bash "$root/glm-validation/run_quartet.sh"
