# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The FP8 online-quant patch must actually reach vLLM's quantized layers.

vLLM-Omni has exactly one place that quantizes activations itself (the MoT
layers).  Every other online-FP8 path -- the per-model ``quantization.py``
modules that swap HF ``nn.Linear`` for a vLLM ``LinearBase`` carrying an
``Fp8Config`` -- quantizes inside vLLM:

    Fp8LinearMethod -> ScaledMMLinearKernel -> self.quant_fp8 (QuantFP8)
      -> QuantFP8.forward_cuda -> ops.scaled_fp8_quant
      -> torch.ops._C.dynamic_per_token_scaled_fp8_quant

``vllm_omni.patch`` redirects that last hop to
``vllm_omni.quantization.fp8_online``.

These tests target the intercept itself: they drive ``QuantFP8`` directly rather
than building a full quantized layer, because the layer path drags in vLLM's
meta-device weight materialisation, which is orthogonal to what the patch
changes.  What must hold is that the dynamic per-token branch routes through
``fp8_online``, that the other branches are left alone, that the result is
unchanged, and that the kill switch still works.  Without this, a vLLM bump that
moves activation quantization off ``QuantFP8`` would silently turn the patch
into a no-op and every per-model FP8 integration would quietly lose the fast
path.
"""

from __future__ import annotations

import os

import pytest
import torch

vllm = pytest.importorskip("vllm")
pytest.importorskip("vllm._custom_ops")

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.quantization.input_quant_fp8 import (  # noqa: E402
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    GroupShape,
)

from vllm_omni.quantization import fp8_online  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the FP8 online fast path requires a CUDA device with sm_80+",
)

ROWS = 128
HIDDEN = 4096


def _disabled() -> bool:
    return (
        fp8_online._DISABLED
        or os.environ.get("VLLM_OMNI_FP8_ONLINE_DISABLE") == "1"
    )


@pytest.fixture
def quant_fp8_enabled():
    """A config that dispatches QuantFP8 to forward_cuda.

    CustomOp.dispatch_forward() resolves ``_forward_method`` at construction
    time and returns **forward_native** when the op is not enabled, only
    returning forward_cuda when it is.  Whether diffusion enables it is exactly
    what #8059 arranges, so a test that does not enable it would silently
    exercise forward_native and prove nothing about this patch.
    """
    cfg = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["+quant_fp8"])
    )
    with set_current_vllm_config(cfg):
        yield


def _quant_fp8(*, static: bool, group_shape: GroupShape) -> QuantFP8:
    return QuantFP8(
        static=static,
        group_shape=group_shape,
        num_token_padding=None,
        compile_native=False,
    )


def _x(dtype=torch.bfloat16):
    torch.manual_seed(0)
    return torch.randn(ROWS, HIDDEN, device="cuda", dtype=dtype) * 2.0


def _count_per_token_calls():
    """Return (calls_dict, restore_fn) around fp8_online.per_token."""
    calls = {"n": 0}
    original = fp8_online.per_token

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    fp8_online.per_token = counting
    return calls, lambda: setattr(fp8_online, "per_token", original)


def test_patch_is_installed():
    """The installer must have replaced QuantFP8.forward_cuda."""
    if _disabled():
        pytest.skip("fast path disabled via VLLM_OMNI_FP8_ONLINE_DISABLE")
    assert getattr(
        QuantFP8.forward_cuda, "_vllm_omni_fp8_online_patched", False
    ), (
        "the FP8 online patch is not installed on QuantFP8.forward_cuda; "
        "vLLM may have moved activation quantization off QuantFP8"
    )


def test_dynamic_per_token_routes_through_fp8_online(quant_fp8_enabled):
    """The dynamic per-token branch must reach fp8_online.per_token."""
    if _disabled():
        pytest.skip("fast path disabled via VLLM_OMNI_FP8_ONLINE_DISABLE")
    if not fp8_online.compiled_ok():
        pytest.skip("CUDA extension unavailable")

    q = _quant_fp8(static=False, group_shape=GroupShape.PER_TOKEN)
    x = _x()
    assert fp8_online.supported(x), "fixture must be inside the fast-path gate"

    calls, restore = _count_per_token_calls()
    try:
        out, scale = q(x)
        torch.accelerator.synchronize()
    finally:
        restore()

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == x.shape
    assert calls["n"] > 0, (
        "QuantFP8.forward_cuda did not route through fp8_online.per_token; "
        "the patch is installed but not reached"
    )


def test_dynamic_per_token_result_is_unchanged(quant_fp8_enabled):
    """Routing through the fast path must not change the quantized result."""
    if not fp8_online.compiled_ok():
        pytest.skip("CUDA extension unavailable")

    q = _quant_fp8(static=False, group_shape=GroupShape.PER_TOKEN)
    x = _x()

    saved = fp8_online._DISABLED
    try:
        fp8_online._DISABLED = True  # reference kernel
        ref_out, ref_scale = q(x)
        torch.accelerator.synchronize()

        fp8_online._DISABLED = False  # fast path
        fast_out, fast_scale = q(x)
        torch.accelerator.synchronize()
    finally:
        fp8_online._DISABLED = saved

    assert torch.equal(
        ref_out.view(torch.uint8), fast_out.view(torch.uint8)
    ), "the patched path changed the FP8 payload"
    assert torch.equal(ref_scale, fast_scale), "the patched path changed the scale"


def test_non_per_token_branch_is_left_alone(quant_fp8_enabled):
    """Per-tensor dynamic must NOT be redirected by the patch.

    The patch claims only the dynamic per-token branch.  If it started
    intercepting this one it would be enabling behaviour with a much narrower
    measured envelope than the caller expects.
    """
    if _disabled():
        pytest.skip("fast path disabled via VLLM_OMNI_FP8_ONLINE_DISABLE")
    if not fp8_online.compiled_ok():
        pytest.skip("CUDA extension unavailable")

    q = _quant_fp8(static=False, group_shape=GroupShape.PER_TENSOR)
    x = _x()

    calls, restore = _count_per_token_calls()
    try:
        out, scale = q(x)
        torch.accelerator.synchronize()
    finally:
        restore()

    assert out.dtype == torch.float8_e4m3fn
    assert calls["n"] == 0, (
        "the patch redirected a non-per-token branch; it must only claim the "
        "dynamic per-token case"
    )


def test_kill_switch_disables_the_redirect(quant_fp8_enabled):
    """With the kill switch on, the reference path must run instead."""
    if not fp8_online.compiled_ok():
        pytest.skip("CUDA extension unavailable")

    q = _quant_fp8(static=False, group_shape=GroupShape.PER_TOKEN)
    x = _x()

    calls, restore = _count_per_token_calls()
    saved = fp8_online._DISABLED
    try:
        fp8_online._DISABLED = True
        out, _scale = q(x)
        torch.accelerator.synchronize()
    finally:
        restore()
        fp8_online._DISABLED = saved

    assert out.dtype == torch.float8_e4m3fn
    assert calls["n"] == 0, (
        "the kill switch did not disable the redirect: fp8_online.per_token "
        "was still called"
    )
