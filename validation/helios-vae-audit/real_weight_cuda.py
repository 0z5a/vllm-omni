"""Real Helios FP32 VAE weights on two CUDA ranks; component parity only."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLWan
from vllm_omni.diffusion.models.helios.autoencoder_kl_wan import HeliosAutoencoderKLWan

from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel


@torch.inference_mode()
def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(ulysses_degree=2)
    model = Path.home() / "vllm-omni-7452-7506-20260915/0z5a/quant-three-20260920/helios/model"
    native = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    candidate = HeliosAutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    candidate.set_parallel_size(2)
    for vae in (native, candidate):
        vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=128,
            tile_sample_stride_width=128,
        )
    generator = torch.Generator(device="cuda").manual_seed(142)
    video = torch.randn(2, 3, 9, 320, 448, device="cuda", generator=generator)
    first = None
    for case, (batch, frames) in enumerate(((1, 1), (1, 5), (1, 9), (2, 5), (1, 1))):
        pixels = video[:batch, :, :frames]
        rng = torch.cuda.get_rng_state()
        expected = native.encode(pixels).latent_dist
        actual = candidate.encode(pixels).latent_dist
        assert torch.equal(expected.parameters, actual.parameters)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        latent = actual.sample(generator=torch.Generator(device="cuda").manual_seed(142))
        reference = expected.sample(generator=torch.Generator(device="cuda").manual_seed(142))
        assert torch.equal(latent, reference)
        expected_video = native.decode(reference, return_dict=False)[0]
        decoded = candidate.decode(latent, return_dict=False)[0]
        assert decoded.shape == pixels.shape and decoded.dtype == torch.float32
        assert torch.isfinite(decoded).all() and torch.equal(decoded, expected_video)
        assert all(cache is None for cache in candidate._feat_map)
        assert all(cache is None for cache in candidate._enc_feat_map)
        if case == 0:
            first = decoded.clone()
        if case == 4:
            assert torch.equal(decoded, first)
        print(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "case": case,
                    "input_shape": list(pixels.shape),
                    "latent_shape": list(latent.shape),
                    "posterior_exact": True,
                    "sample_exact": True,
                    "rng_unchanged": True,
                    "decode_exact": True,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
