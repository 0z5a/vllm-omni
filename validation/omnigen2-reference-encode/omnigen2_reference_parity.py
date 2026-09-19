"""Real-weight multi-reference encode correctness, not full image-edit E2E."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKL

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm_omni.diffusion.models.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(ulysses_degree=2)
    path = Path.home() / "omnigen2-encoder-fp8-20260915/model"
    native = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    candidate = DistributedAutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    native.enable_tiling()
    candidate.enable_tiling()
    candidate.set_parallel_size(2)
    reference_pipeline = OmniGen2Pipeline.__new__(OmniGen2Pipeline)
    torch.nn.Module.__init__(reference_pipeline)
    reference_pipeline.vae = native
    reference_pipeline.vae_scale_factor = 8
    pipeline = OmniGen2Pipeline.__new__(OmniGen2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae = candidate
    pipeline.vae_scale_factor = 8
    with torch.inference_mode():
        for shapes in (((512, 512),), ((512, 768), (768, 512)), ((512, 512), (384, 768), (768, 384)), ((512, 512),)):
            generator = torch.Generator(device=f"cuda:{rank}").manual_seed(321)
            images = [torch.randn(1, 3, h, w, device="cuda", dtype=torch.bfloat16, generator=generator) for h, w in shapes]
            torch.manual_seed(142)
            expected = [reference_pipeline.encode_vae(image).squeeze(0) for image in images]
            expected_rng = torch.cuda.get_rng_state()
            torch.manual_seed(142)
            actual = pipeline._encode_reference_images(images)
            assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            print(json.dumps({"rank": rank, "image_shapes": shapes, "exact": True, "rng_equal": True}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
