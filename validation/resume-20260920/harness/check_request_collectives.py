"""Two-rank CUDA correctness probe; this is not full-model E2E evidence."""

import json
import os
from functools import partial
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from vllm.config import VllmConfig, set_current_vllm_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.hsdp import (
    HSDPInferenceConfig,
    apply_hsdp_to_model,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image import OvisImagePipeline
from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import (
    OvisImageTransformer2DModel,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(cfg_parallel_size=2)
    torch.manual_seed(142)
    torch.set_default_dtype(torch.bfloat16)
    config = OmniDiffusionConfig(
        model="/home/kxqandccx/omni-1217-20260919/model", tf_model_config=TransformerConfig(params={"num_layers": 1})
    )
    build = partial(
        OvisImageTransformer2DModel,
        config,
        in_channels=16,
        out_channels=16,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(2, 2, 4),
    )
    model = build().eval().to(rank)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.02)
    reference = build().eval().to(rank)
    reference.load_state_dict(model.state_dict())
    sharded = apply_hsdp_to_model(
        model,
        HSDPInferenceConfig(enabled=True, hsdp_shard_size=2),
        target_device=torch.device("cuda", rank),
    )
    assert sum(isinstance(module, FSDPModule) for module in sharded.modules()) == 3
    assert all(isinstance(parameter, DTensor) for parameter in sharded.parameters())
    pipeline = OvisImagePipeline.__new__(OvisImagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = sharded
    pipeline.od_config = config
    config.additional_config["hsdp_reshard_after_forward"] = False
    with torch.no_grad():
        for sequence in (16, 24, 16):
            inputs = {
                "hidden_states": torch.randn(1, sequence, 16, device=rank),
                "encoder_hidden_states": torch.randn(1, 10, 32, device=rank),
                "timestep": torch.tensor([1], device=rank),
                "img_ids": torch.zeros(sequence, 3, device=rank),
                "txt_ids": torch.zeros(10, 3, device=rank),
                "return_dict": False,
            }
            expected = reference(**inputs)[0]
            with patch.object(dist, "all_gather_single", wraps=dist.all_gather_single) as gather:
                with pipeline._hsdp_denoising_context():
                    for _ in range(3):
                        actual = sharded(**inputs)[0]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert all(not isinstance(p, DTensor) for p in sharded.parameters())
                assert gather.call_count == 3, gather.call_count
            assert all(isinstance(p, DTensor) for p in sharded.parameters())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "sequence": sequence,
                        "max_abs": (actual - expected).abs().max().item(),
                        "all_gathers_for_three_steps": gather.call_count,
                    }
                ),
                flush=True,
            )
    try:
        with torch.no_grad(), pipeline._hsdp_denoising_context():
            sharded(**inputs)
            raise RuntimeError("injected after first step")
    except RuntimeError as error:
        assert str(error) == "injected after first step"
    assert all(isinstance(parameter, DTensor) for parameter in sharded.parameters())
    print(json.dumps({"rank": rank, "exception_cleanup": "passed"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig()):
        main()
