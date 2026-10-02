"""Inspect real TP1 online loaders on CPU with only CUDA quantization mocked."""

import json
import site
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(ROOT / "dependencies"))

import torch
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.online import fp8 as online
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader


def quantize(weight, scale):
    return (weight / scale).to(torch.float8_e4m3fn), scale


def main():
    kernel = SimpleNamespace(process_weights_after_loading=lambda layer: None)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    with (
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch.object(online, "get_current_vllm_config", return_value=config),
        patch.object(online, "cutlass_fp8_supported", return_value=True),
        patch.object(online, "init_fp8_linear_kernel", return_value=kernel),
        patch.object(fp8, "get_marlin_input_dtype", return_value=None),
        patch.object(online.ops, "scaled_fp8_quant", side_effect=quantize),
    ):
        quant_config = fp8.Fp8Config()
        model = torch.nn.ModuleDict(
            {
                "column": linear.ColumnParallelLinear(
                    4, 8, quant_config=quant_config, params_dtype=torch.bfloat16
                ),
                "row": linear.RowParallelLinear(
                    4, 8, quant_config=quant_config, params_dtype=torch.bfloat16
                ),
                "qkv": linear.QKVParallelLinear(
                    4, 2, 2, quant_config=quant_config, params_dtype=torch.bfloat16
                ),
                "replicated": linear.ReplicatedLinear(
                    4, 8, quant_config=quant_config, params_dtype=torch.bfloat16
                ),
            }
        )
        weights = []
        for name in model:
            if name == "qkv":
                weights.extend(
                    (f"qkv.bias.{shard}", torch.ones(4)) for shard in ("q", "k", "v")
                )
                weights.extend(
                    (f"qkv.weight.{shard}", torch.ones(4, 4))
                    for shard in ("q", "k", "v")
                )
            else:
                weights.extend(
                    [
                        (f"{name}.bias", torch.ones(8)),
                        (f"{name}.weight", torch.ones(8, 4)),
                    ]
                )
        states = []
        for name, value in DiffusersPipelineLoader._stream_online_quant_weights_to_cpu(
            model, weights
        ):
            parts = name.split(".")
            layer = model[parts[0]]
            parameter = dict(layer.named_parameters())[parts[1]]
            parameter.weight_loader(parameter, value, parts[2]) if len(
                parts
            ) == 3 else parameter.weight_loader(parameter, value)
            info = get_layerwise_info(layer)
            state = {
                "weight": name,
                "load_numel": info.load_numel,
                "load_numel_total": info.load_numel_total,
                "dtype": str(layer.weight.dtype),
                "device": str(layer.weight.device),
            }
            states.append(state)
            print(json.dumps(state), flush=True)
        (ROOT / "evidence/online-linear-cpu-diagnostic-v1.json").write_text(
            json.dumps(states, indent=2)
        )


if __name__ == "__main__":
    main()
