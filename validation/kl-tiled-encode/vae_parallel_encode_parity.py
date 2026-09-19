"""Check real VAE tiled encode posterior and RNG on WORLD=2; not full-model E2E."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import DistributedAutoencoderKLFlux2
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(ulysses_degree=2)
    home = Path.home()
    cases = (
        ("ernie", home / "omni-1217-20260919/ernie/model", AutoencoderKLFlux2, DistributedAutoencoderKLFlux2),
        ("omnigen2", home / "omnigen2-encoder-fp8-20260915/model", AutoencoderKL, DistributedAutoencoderKL),
    )
    with torch.inference_mode():
        for name, path, native_class, distributed_class in cases:
            native = native_class.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
            candidate = distributed_class.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
            native.enable_tiling()
            candidate.enable_tiling()
            candidate.set_parallel_size(2)
            torch.manual_seed(142)
            for height, width in ((512, 512), (1152, 1024), (512, 512)):
                pixels = torch.randn(1, 3, height, width, device="cuda", dtype=torch.bfloat16)
                rng_before = torch.cuda.get_rng_state()
                expected = native.encode(pixels).latent_dist
                actual = candidate.encode(pixels).latent_dist
                assert torch.equal(torch.cuda.get_rng_state(), rng_before)
                delta = (actual.parameters.float() - expected.parameters.float()).abs()
                torch.testing.assert_close(actual.parameters, expected.parameters, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual.sample(generator=torch.Generator(device="cuda").manual_seed(42)),
                    expected.sample(generator=torch.Generator(device="cuda").manual_seed(42)),
                    rtol=0, atol=0,
                )
                print(json.dumps({"model": name, "rank": rank, "image_shape": list(pixels.shape), "max_abs": delta.max().item(), "exact": True, "rng_unchanged": True, "strategy": "tile" if max(height, width) > native.tile_sample_min_size else "native"}), flush=True)
            del native, candidate, pixels, expected, actual
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
