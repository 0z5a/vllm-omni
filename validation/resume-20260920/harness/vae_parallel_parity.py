"""Measure real VAE decode parity on WORLD=2; not full-model E2E."""

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
            for height, width in ((64, 64), (128, 144), (64, 64)):
                latent = torch.randn(1, native.config.latent_channels, height, width, device="cuda", dtype=torch.bfloat16)
                expected = native.decode(latent, return_dict=False)[0]
                actual = candidate.decode(latent, return_dict=False)[0]
                if rank == 0:
                    assert actual.shape == expected.shape and torch.isfinite(actual).all()
                    delta = (actual.float() - expected.float()).abs()
                    print(json.dumps({"model": name, "latent_shape": list(latent.shape), "max_abs": delta.max().item(), "mean_abs": delta.mean().item(), "relative_l2": (delta.norm() / expected.float().norm()).item(), "strategy": "tile" if max(height, width) > native.tile_latent_min_size else "patch"}), flush=True)
            del native, candidate, latent, expected, actual
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
