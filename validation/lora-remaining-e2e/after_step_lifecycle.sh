#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail
root=/home/kxqandccx/omni-1217-20260919
while ps -p 1999496 -o args= | rg -Fxq "bash $root/step-lifecycle-validation/after_lora_parallel.sh"; do sleep 30; done
test "$(cat "$root/step-lifecycle-validation/evidence/matrix.status")" = complete
cd "$root/lora-remaining-validation"
bash run_remaining.sh
