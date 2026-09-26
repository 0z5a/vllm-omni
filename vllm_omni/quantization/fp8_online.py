# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA fast path for FP8 E4M3 online activation quantization (sm_80+).

Why this module exists
----------------------
``vllm_omni``'s diffusion MoT layers quantize activations online with

    ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)

which lands in vLLM's ``dynamic_per_token_scaled_fp8_quant``.  That path is
correct but pays for a generic implementation.  This module adds a narrow,
auditable CUDA fast path with an explicit architecture gate and an explicit
fallback to the original implementation.

Coverage (the fast path is selected only when *all* of these hold):

* CUDA device with compute capability >= 8.0 (sm_80, sm_86, sm_89, sm_90,
  sm_100, sm_120 all build and run; see ``ARCH_EXTENSION_NAME``);
* input dtype ``bfloat16`` / ``float16`` / ``float32``;
* 2-D contiguous activation with a row length divisible by 4;
* dynamic (``scale is None``) quantization, per-token or per-tensor;
* an optional one-element fp32 CUDA ``scale_ub``.

Everything else is delegated to the reference implementation, so no caller
observes a behaviour change outside that envelope.

Numeric contract
----------------
Frozen against the reference kernel; see ``docs/fp8_online_contract.md``.  The
short version:

    amax      = max_j |x[r, j]|
    scale[r]  = min( (double)amax / 448.0, 1/(448*512) )   -> fp32
    out[r, j] = e4m3_rn_sat( x[r, j] / scale[r] )

The scale divide happens in double precision on purpose: doing it in fp32
changes roughly half the scale values by 1 ULP and therefore changes payload
bytes, which would silently alter decoding.
"""

from __future__ import annotations

import os
import threading

# torch.utils.cpp_extension resolves the CUDA toolkit directory once, when that
# module is first imported, from CUDA_HOME/CUDA_PATH or by finding nvcc on PATH.
# A serving process may inherit neither, so pin CUDA_HOME here -- before torch is
# imported below -- otherwise the extension build fails on missing
# cuda_runtime_api.h even though the toolkit is installed.
if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
    for _candidate in ("/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-12.8"):
        if os.path.isfile(os.path.join(_candidate, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = _candidate
            break

import torch

__all__ = [
    "ARCH_EXTENSION_NAME",
    "FASTPATH_ARCH_FLOOR",
    "compiled_ok",
    "per_tensor",
    "per_token",
    "scaled_fp8_quant",
    "supported",
]

# sm_80 = Ampere.  This is the floor the kernels are compiled and guarded for.
FASTPATH_ARCH_FLOOR = (8, 0)

#: Largest row count for which the fast path is selected.
#:
#: Measured on RTX 5090 (sm_120), bf16 in / e4m3 out, K in {3072, 4096}, GPU
#: warmed to a stable clock, arms interleaved.  Within this bound the fast path
#: wins by 1.13x-1.26x in every measured configuration.  Above it the reference
#: kernel is better or equal -- at M=1024/K=4096 it is 0.90x, and at M>=2048 it
#: reaches 0.56x, because the reference parallelises a long row across more
#: blocks than this implementation does.  Since the gate has to hold for *every*
#: row count it admits, it stops at the largest bound that was verified
#: uniformly.  See docs/fp8_online_results.md for the full table.
FASTPATH_MAX_ROWS = 512

#: Compiled code objects shipped in the extension, for auditability.
ARCH_EXTENSION_NAME = "sm_80;sm_86;sm_89;sm_90;sm_100;sm_120"

_EXTENSION_NAME = "_vllm_omni_fp8_online_quant"

_ext = None
_ext_lock = threading.Lock()
_ext_unavailable_logged = False

# Architecture list used for the first build.  Kept explicit (rather than
# "native") so one build covers every guarded target and the intent is visible
# in build logs.  Only the sm_80 entry is the floor; the rest are shipped so a
# single wheel/extension works across the fleet.
_DEFAULT_ARCH_LIST = "8.0;8.6;8.9;9.0;10.0;12.0"

#: Candidate locations for the CUDA sources, relative to this file.  The first
#: one is the in-tree layout (``vllm_omni/quantization/csrc``), the second keeps
#: working when the sources sit next to the module (experiment checkouts).
_CSRC_CANDIDATES = ("csrc", os.path.join("..", "csrc"), os.path.join("..", "..", "csrc"))


def _find_csrc() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in _CSRC_CANDIDATES:
        cand = os.path.normpath(os.path.join(here, rel))
        if os.path.isfile(os.path.join(cand, "fp8_online_quant_bindings.cu")):
            return cand
    return None


def _load_extension():
    """Build/load the CUDA extension once; return None when unavailable."""
    global _ext, _ext_unavailable_logged
    if _ext is not None:
        return _ext or None
    with _ext_lock:
        if _ext is not None:
            return _ext or None
        try:
            if not torch.cuda.is_available():
                _ext = False
                return None

            csrc = _find_csrc()
            if csrc is None:
                raise FileNotFoundError(
                    "fp8_online: CUDA sources not found next to "
                    f"{os.path.abspath(__file__)}"
                )

            from torch.utils.cpp_extension import load

            # CUDA_HOME is resolved (and cached) by torch.utils.cpp_extension at
            # its own import time, which happens while torch initialises -- too
            # early for this module to influence it.  Pass the toolkit paths
            # explicitly so the build works regardless of what the service
            # process inherited.  libraries are named without the "lib" prefix so
            # this also works on Windows; the CUDA runtime ships with the driver
            # on Linux, so only the toolkit dirs matter for headers and stubs.
            cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
            include_dirs = [
                os.path.join(cuda_home, "include"),
                os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
            ]
            library_dirs = [
                os.path.join(cuda_home, "lib64"),
                os.path.join(cuda_home, "targets", "x86_64-linux", "lib"),
            ]
            include_dirs = [d for d in include_dirs if os.path.isdir(d)]
            library_dirs = [d for d in library_dirs if os.path.isdir(d)]
            if not include_dirs:
                raise FileNotFoundError(
                    f"fp8_online: no CUDA include directory found under {cuda_home}"
                )

            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _DEFAULT_ARCH_LIST)
            _ext = load(
                name=_EXTENSION_NAME,
                sources=[
                    os.path.join(csrc, "fp8_online_quant_bindings.cu"),
                ],
                extra_include_paths=[csrc] + include_dirs,
                extra_ldflags=[f"-L{d}" for d in library_dirs],
                extra_cuda_cflags=[
                    "-O3",
                    # Keep IEEE semantics: the reference path does not use fast
                    # math and the contract depends on exact fp32 division.
                    "--ftz=false",
                    "--prec-div=true",
                    "--prec-sqrt=true",
                ],
                verbose=os.environ.get("VLLM_OMNI_FP8_ONLINE_VERBOSE") == "1",
            )
            return _ext
        except Exception:
            if not _ext_unavailable_logged:
                _ext_unavailable_logged = True
                if os.environ.get("VLLM_OMNI_FP8_ONLINE_VERBOSE") == "1":
                    import traceback

                    traceback.print_exc()
            _ext = False
            return None


def compiled_ok() -> bool:
    """True when the CUDA extension is built and importable."""
    return _load_extension() is not None


_capability_cache: dict[int, tuple[int, int]] = {}

#: Kill switch.  ``VLLM_OMNI_FP8_ONLINE_DISABLE=1`` forces every call onto the
#: reference implementation.  It exists so an A/B comparison can toggle the fast
#: path inside one process (identical weights, allocator, and graph state) and so
#: an operator can disable the fast path without a code change.
_DISABLED = os.environ.get("VLLM_OMNI_FP8_ONLINE_DISABLE") == "1"


def _device_capability(index: int) -> tuple[int, int]:
    """Cached compute capability; querying it per call is measurable overhead."""
    cap = _capability_cache.get(index)
    if cap is None:
        cap = torch.cuda.get_device_capability(index)
        _capability_cache[index] = cap
    return cap


def supported(x: torch.Tensor) -> bool:
    """Whether the fast path can handle ``x`` for dynamic quantization.

    This is a pure predicate: it never builds the extension and never touches a
    device, so it is safe to call from dispatch code on any platform.
    """
    if _DISABLED:
        return False
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    if x.dim() != 2:
        return False
    if not x.is_contiguous():
        return False
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if x.shape[1] == 0 or x.shape[1] % 4 != 0:
        return False
    if x.shape[0] == 0 or x.shape[0] > FASTPATH_MAX_ROWS:
        return False
    if _device_capability(x.get_device()) < FASTPATH_ARCH_FLOOR:
        return False
    return True


def _reference_per_token(x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor,
                         scale_ub: torch.Tensor | None) -> None:
    getattr(torch.ops._C, "dynamic_per_token_scaled_fp8_quant")(
        out, x, scale, scale_ub
    )


def _reference_per_tensor(x: torch.Tensor, out: torch.Tensor,
                          scale: torch.Tensor) -> None:
    getattr(torch.ops._C, "dynamic_scaled_fp8_quant")(out, x, scale)


def per_token(out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor,
              scale_ub: torch.Tensor | None = None) -> None:
    """Per-token dynamic FP8 quantization into preallocated ``out``/``scale``.

    Falls back to the reference kernel whenever the fast path does not apply.
    """
    ext = _load_extension() if supported(x) else None
    if ext is None:
        _reference_per_token(x, out, scale, scale_ub)
        return
    ext.fp8_online_per_token_quant(out, x, scale, scale_ub)


def per_tensor(out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor,
               scale_ub: torch.Tensor | None = None) -> None:
    """Per-tensor dynamic FP8 quantization into preallocated ``out``/``scale``."""
    if supported(x) and (scale_ub is None or scale_ub.numel() == 1):
        ext = _load_extension()
        if ext is not None:
            ext.fp8_online_per_tensor_quant(out, x, scale, scale_ub)
            return
    _reference_per_tensor(x, out, scale)


def scaled_fp8_quant(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``vllm._custom_ops.scaled_fp8_quant``.

    Only the dynamic branch (``scale is None``) is accelerated.  Static
    quantization, padding, and any unsupported layout go straight to vLLM, so
    the returned tensors keep exactly the reference semantics.
    """
    if scale is not None or num_token_padding is not None:
        from vllm import _custom_ops as ops

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
        )

    if not supported(x):
        from vllm import _custom_ops as ops

        return ops.scaled_fp8_quant(
            x,
            None,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
        )

    out_dtype = torch.float8_e4m3fn
    m, n = x.shape
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    if use_per_token_if_dynamic:
        s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
        per_token(out, x, s, scale_ub)
    else:
        s = torch.empty(1, device=x.device, dtype=torch.float32)
        per_tensor(out, x, s, scale_ub)
    return out, s
