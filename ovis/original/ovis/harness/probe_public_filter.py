"""Check the existing vLLM kernel exclusion before CUDA initialization."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
from vllm.model_executor.kernels.linear import is_supported_and_can_implement_kernel
from vllm.model_executor.kernels.linear.scaled_mm import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)

assert not torch.cuda.is_initialized()
original = CutlassFP8ScaledMMLinearKernel.is_supported.__func__
assert original.__module__ == CutlassFP8ScaledMMLinearKernel.__module__
config = FP8ScaledMMLinearLayerConfig(
    weight_quant_key=kFp8StaticTensorSym,
    activation_quant_key=kFp8DynamicTensorSym,
    input_dtype=torch.bfloat16,
    out_dtype=torch.bfloat16,
    weight_shape=(4096, 4096),
)
supported, reason = is_supported_and_can_implement_kernel(
    CutlassFP8ScaledMMLinearKernel, config, 80
)
assert not supported and "disabled by environment variable" in reason
assert CutlassFP8ScaledMMLinearKernel.is_supported.__func__ is original
assert not torch.cuda.is_initialized()
paths = [
    Path(inspect.getfile(is_supported_and_can_implement_kernel)),
    Path(inspect.getfile(CutlassFP8ScaledMMLinearKernel)),
]
print(
    json.dumps(
        {
            "public_control": "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel",
            "disabled_kernel_rejected": True,
            "class_method_unchanged": True,
            "cuda_initialized": torch.cuda.is_initialized(),
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "limits": "CPU filter probe only; real model selection, native kernels and image checks remain pending.",
        },
        indent=2,
    )
)
