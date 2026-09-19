#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919/rebased
for attempt in $(seq 1 180); do
    if nvidia-smi -i 2,3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if ($1 > 2000 || $2 > 5) busy=1} END {exit busy}'; then
        date -Is
        exec bash "$root/run_validation.sh"
    fi
    sleep 10
done
printf "GPU 2,3 still occupied; no engine started.\n"
exit 75
