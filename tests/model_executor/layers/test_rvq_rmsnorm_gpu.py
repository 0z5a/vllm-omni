# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU regression coverage for both observable RVQ/RMSNorm outputs."""

import unittest
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from vllm_omni.platforms.interface import OmniPlatform

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


class GpuRmsnormTests(unittest.TestCase):
    torch: ClassVar[ModuleType]
    variant: ClassVar[ModuleType]
    native: ClassVar[Callable]
    platform: ClassVar[type["OmniPlatform"]]
    budgets: ClassVar[tuple[int, int]]
    compiled: ClassVar[Callable]

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from vllm.triton_utils import triton
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("requires an SM120 GPU")
        if triton.__version__ != "3.7.1":
            raise unittest.SkipTest("requires the audited Triton 3.7.1 launcher ABI")
        import torch._dynamo.config as config
        import torch._inductor.config as inductor_config

        from benchmarks.kernels import rvq_rmsnorm
        from benchmarks.kernels.rvq_rmsnorm import native_region
        from vllm_omni.platforms import current_omni_platform

        if not inductor_config.emulate_precision_casts:
            raise RuntimeError("Set TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 before starting pytest")
        cls.torch, cls.variant, cls.native = torch, rvq_rmsnorm, staticmethod(native_region)
        cls.platform = current_omni_platform
        cls.budgets = (config.recompile_limit, config.accumulated_recompile_limit)
        config.recompile_limit = 256
        config.accumulated_recompile_limit = max(1024, config.accumulated_recompile_limit)
        cls.compiled = staticmethod(torch.compile(native_region, fullgraph=True, dynamic=False))

    @classmethod
    def tearDownClass(cls):
        import torch._dynamo.config as config

        config.recompile_limit, config.accumulated_recompile_limit = cls.budgets

    def inputs(self, dtype, *, hidden=37, strided=False, index_dtype=None, seed=42):
        torch = self.torch
        index_dtype = torch.int64 if index_dtype is None else index_dtype
        generator = torch.Generator(device="cuda").manual_seed(seed)
        backing = torch.randint(32, (2, 16, 7 if strided else 3), device="cuda", dtype=index_dtype, generator=generator)
        codes = backing[..., 1::2] if strided else backing
        weight = torch.randn(16 * 32, hidden, device="cuda", dtype=dtype, generator=generator)
        offsets = (torch.arange(16, device="cuda", dtype=index_dtype) * 32).view(1, 16, 1)
        gamma = torch.randn(hidden, device="cuda", dtype=dtype, generator=generator)
        if strided:
            storage = torch.empty((1, 33, 1), device="cuda", dtype=index_dtype)
            offset_view = storage[:, 1::2, :]
            offset_view.copy_(offsets)
            offsets = offset_view
            storage = torch.empty(2 * hidden + 1, device="cuda", dtype=dtype)
            gamma_view = storage[1::2]
            gamma_view.copy_(gamma)
            gamma = gamma_view
        return codes, weight, offsets, gamma, 1e-5

    def assert_pair(self, actual, expected):
        torch = self.torch
        self.assertEqual(type(actual), tuple)
        self.assertEqual(len(actual), 2)
        if actual[0].numel():
            self.assertNotEqual(actual[0].data_ptr(), actual[1].data_ptr())
        for value, target in zip(actual, expected):
            self.assertEqual(value.shape, target.shape)
            self.assertEqual(value.dtype, target.dtype)
            for classifier in (torch.isnan, torch.isposinf, torch.isneginf, torch.isfinite):
                self.assertTrue(torch.equal(classifier(value), classifier(target)))
            finite = torch.isfinite(target)
            torch.testing.assert_close(value[finite], target[finite], atol=0.002, rtol=0.002)

    def check(self, call, inputs):
        expected = self.native(*inputs)
        compiled = self.compiled(*inputs)
        actual = call(*inputs)
        self.assert_pair(compiled, expected)
        self.assert_pair(actual, expected)
        self.assert_pair(actual, compiled)
        return actual

    def test_native_order_raw_and_norm_on_frozen_runtime(self):
        torch = self.torch
        if torch.__version__ != "2.13.0+cu130":
            self.skipTest("Bit-exact reduction contract was measured on Torch 2.13.0+cu130")
        call = self.variant.make_variant(4)
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                for frames in (1, 32, 128):
                    for strided in (False, True):
                        generator = torch.Generator(device="cuda").manual_seed(947)
                        backing = torch.randint(2048, (1, 16, frames * 2), device="cuda", generator=generator)
                        codes = backing[..., ::2] if strided else backing[..., :frames].contiguous()
                        weight = torch.randn(32768, 1024, dtype=dtype, device="cuda", generator=generator)
                        offsets = (torch.arange(16, device="cuda") * 2048).view(1, 16, 1)
                        gamma = torch.randn(1024, dtype=dtype, device="cuda", generator=generator)
                        for delta in (0, 17):
                            if delta:
                                codes.copy_((codes + delta) % 2048)
                            inputs = codes, weight, offsets, gamma, 1e-5
                            for actual, expected in zip(call(*inputs), self.native(*inputs), strict=True):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_both_configurations_dtypes_hidden_tail_strides_and_new_buffers(self):
        torch = self.torch
        with torch.inference_mode():
            for warps in (4, 8):
                call = self.variant.make_variant(warps)
                keep_alive = []
                for dtype in (torch.float16, torch.bfloat16):
                    for hidden in (37, 1024):
                        for strided in (False, True):
                            with self.subTest(warps=warps, dtype=dtype, hidden=hidden, strided=strided):
                                for seed in (42, 43):
                                    inputs = self.inputs(dtype, hidden=hidden, strided=strided, seed=seed)
                                    keep_alive.append(inputs)
                                    self.check(call, inputs)
                                self.check(call, inputs)

    def test_nonfinite_values_zero_epsilon_and_fp32_variance_overflow(self):
        torch = self.torch
        with torch.inference_mode():
            for warps in (4, 8):
                call = self.variant.make_variant(warps)
                for dtype in (torch.float16, torch.bfloat16):
                    for pattern in (
                        "nan",
                        "positive_inf",
                        "negative_inf",
                        "opposite_inf",
                        "gamma_nonfinite",
                        "zero_eps",
                    ):
                        with self.subTest(warps=warps, dtype=dtype, pattern=pattern):
                            inputs = list(self.inputs(dtype))
                            inputs[0].zero_()
                            if pattern == "nan":
                                inputs[1][0, 0] = float("nan")
                            elif pattern == "positive_inf":
                                inputs[1][0, 0] = float("inf")
                            elif pattern == "negative_inf":
                                inputs[1][0, 0] = -float("inf")
                            elif pattern == "opposite_inf":
                                inputs[1][0, 0], inputs[1][32, 0] = float("inf"), -float("inf")
                            elif pattern == "gamma_nonfinite":
                                inputs[3][0], inputs[3][1] = float("inf"), float("nan")
                            else:
                                inputs[1].zero_()
                                inputs[4] = 0.0
                            self.check(call, tuple(inputs))
                inputs = list(self.inputs(torch.bfloat16))
                inputs[1].fill_(1e20)  # finite raw; separately fp32 square overflows
                self.check(call, tuple(inputs))

    def test_cached_nondefault_stream_and_graph_replay_use_live_gamma_codes_and_epsilon(self):
        torch = self.torch
        with torch.inference_mode():
            call = self.variant.make_variant(8)
            inputs = self.inputs(torch.bfloat16, hidden=1024, strided=True)
            self.check(call, inputs)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(2_000_000)
                inputs[0].fill_(4)
                inputs[3].add_(0.25)
                output = call(*inputs)
            stream.synchronize()
            self.assert_pair(output, self.native(*inputs))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call(*inputs)
            for index in (1, 5, 9):
                inputs[0].fill_(index)
                inputs[3].add_(0.0625)
                graph.replay()
                self.platform.synchronize()
                self.assert_pair(captured, self.native(*inputs))
            changed_epsilon = (*inputs[:-1], 0.125)
            self.check(call, changed_epsilon)

    def test_fallback_promotion_lazy_views_autograd_empty_and_zero_stride_gamma(self):
        torch = self.torch
        call = self.variant.make_variant(4)
        with torch.inference_mode():
            inputs = self.inputs(torch.bfloat16, index_dtype=torch.int32)
            self.check(call, inputs)
            for dtype in (torch.float16, torch.float32, torch.float64):
                changed = (*inputs[:3], inputs[3].to(dtype), inputs[4])
                self.assertFalse(self.variant.can_use_variant(*changed))
                self.assert_pair(call(*changed), self.native(*changed))
                self.assertEqual(call(*changed)[1].dtype, torch.promote_types(torch.bfloat16, dtype))
            for index in range(4):
                changed_inputs = list(inputs)
                # Integer logical indices stay valid while retaining a lazy
                # negative view bit, so this test cannot poison the CUDA context.
                changed_inputs[index] = torch._neg_view(-changed_inputs[index])
                self.assertFalse(self.variant.can_use_variant(*changed_inputs))
                self.assert_pair(call(*changed_inputs), self.native(*changed_inputs))
            broadcast_gamma = inputs[3][:1].expand(inputs[3].shape[0])
            self.check(call, (*inputs[:3], broadcast_gamma, inputs[4]))
            empty = (inputs[0][:0], *inputs[1:])
            self.assert_pair(call(*empty), self.native(*empty))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.assertFalse(self.variant.can_use_variant(*inputs))
                self.assert_pair(call(*inputs), self.native(*inputs))
        with torch.enable_grad():
            inputs = list(self.inputs(torch.float16))
            inputs[1].requires_grad_()
            inputs[3].requires_grad_()
            self.assertFalse(self.variant.can_use_variant(*inputs))
            raw, normalized = call(*inputs)
            (raw.float().sum() + normalized.float().sum()).backward()
            self.assertIsNotNone(inputs[1].grad)
            self.assertIsNotNone(inputs[3].grad)


if __name__ == "__main__":
    unittest.main()
