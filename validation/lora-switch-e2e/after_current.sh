#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 681180 -o args= | rg -Fq "$root/resume_long_hsdp.sh"; do sleep 30; done
test "$(cat "$root/compat-validation/evidence/nohit.status")" = complete
cd "$root/lora-switch-validation"
sha256sum -c source.sha256
bash run_cache_cases.sh
