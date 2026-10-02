#!/usr/bin/env bash
# Compile-evidence matrix for the FP8 online kernels.
#
# Compiles the dependency-free kernel translation unit (tests/fp8_online_harness.cu
# includes csrc/fp8_online_quant.cuh and instantiates every kernel
# specialisation) for each architecture the runtime guard admits, and records
# PTX, SASS and per-kernel register/spill usage.
#
# Compiling for a target is NOT evidence that it runs.  This script only ever
# claims COMPILE_ONLY for targets other than the one it executes on.
set -uo pipefail

ROOT="${ROOT:-$HOME/0z5a/work/fp8-online-quant}"
if [ -d "$ROOT/vllm_omni/quantization/csrc" ]; then
  CSRC="$ROOT/vllm_omni/quantization/csrc"
else
  CSRC="$ROOT/csrc"
fi
TU="$ROOT/tests/fp8_online_harness.cu"
OUT="$ROOT/results/compile"
mkdir -p "$OUT"

# Same numeric flags the runtime loader uses, so the evidence matches what ships.
FLAGS=(-std=c++17 -O3 -lineinfo --ftz=false --prec-div=true --prec-sqrt=true
       -I"$CSRC" -I"$ROOT/tests")

SM_LIST=(80 86 89 90 100 120)
STATUS="$OUT/status.txt"
: > "$STATUS"

for sm in "${SM_LIST[@]}"; do
  log="$OUT/sm${sm}.log"
  if nvcc "${FLAGS[@]}" -arch="sm_${sm}" -cubin "$TU" \
        -o "$OUT/kernels_sm${sm}.cubin" > "$log" 2>&1; then
    nvcc "${FLAGS[@]}" -arch="sm_${sm}" -ptx "$TU" \
        -o "$OUT/kernels_sm${sm}.ptx" >> "$log" 2>&1
    nvcc "${FLAGS[@]}" -arch="sm_${sm}" -Xptxas=-v -cubin "$TU" \
        -o /dev/null >> "$log" 2>&1
    cuobjdump --dump-sass "$OUT/kernels_sm${sm}.cubin" > "$OUT/sass_sm${sm}.txt" 2>/dev/null
    conv=$(grep -ohE "F2FP[A-Z0-9._]*|CVT[A-Z0-9._]*" "$OUT/sass_sm${sm}.txt" 2>/dev/null | sort -u | tr '\n' ' ')
    red=$(grep -ohE "REDUX[A-Z0-9._]*|SHFL[A-Z0-9._]*" "$OUT/sass_sm${sm}.txt" 2>/dev/null | sort -u | tr '\n' ' ')
    echo "sm_${sm}: COMPILE_PASS; runtime UNTESTED | convert: ${conv:-none} | reduce: ${red:-none}" \
      | tee -a "$STATUS"
  else
    echo "sm_${sm}: COMPILE_FAIL" | tee -a "$STATUS"
    tail -12 "$log"
  fi
done

echo
echo "=== sm_120 per-kernel resources (registers / spills) ==="
grep -E "Compiling entry|Used [0-9]+ registers|spill stores" "$OUT/sm120.log" \
  | sed -e 's/_ZN13vllm_omni_fp8//' -e 's/EEv.*//' -e 's/ptxas info *: *//' \
  | paste - - - 2>/dev/null | head -30
