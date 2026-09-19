"""Real Helios VAE weights on two CPU/Gloo ranks; no GPU speed claim."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLWan
from vllm_omni.diffusion.models.helios.autoencoder_kl_wan import HeliosAutoencoderKLWan


@torch.inference_mode()
def main() -> None:
    dist.init_process_group("gloo")
    model = Path.home() / "vllm-omni-7452-7506-20260915/0z5a/quant-three-20260920/helios/model"
    native = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32).eval()
    with patch(
        "vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor.get_world_group",
        return_value=SimpleNamespace(device_group=dist.group.WORLD),
    ):
        candidate = HeliosAutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32).eval()
    candidate.set_parallel_size(2)
    for vae in (native, candidate):
        vae.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=16,
            tile_sample_stride_width=16,
        )
    generator = torch.Generator().manual_seed(142)
    video = torch.randn(1, 3, 9, 48, 64, generator=generator)
    first = None
    for case, frames in enumerate((1, 5, 9, 1)):
        pixels = video[:, :, :frames]
        rng = torch.random.get_rng_state()
        expected = native.encode(pixels).latent_dist
        actual = candidate.encode(pixels).latent_dist
        assert torch.equal(expected.parameters, actual.parameters)
        assert torch.equal(rng, torch.random.get_rng_state())
        latent = actual.sample(generator=torch.Generator().manual_seed(142))
        reference = expected.sample(generator=torch.Generator().manual_seed(142))
        assert torch.equal(latent, reference)
        expected_video = native.decode(reference, return_dict=False)[0]
        decoded = candidate.decode(latent, return_dict=False)[0]
        assert decoded.shape == pixels.shape and decoded.dtype == torch.float32
        assert torch.isfinite(decoded).all() and torch.equal(decoded, expected_video)
        assert all(cache is None for cache in candidate._feat_map)
        assert all(cache is None for cache in candidate._enc_feat_map)
        if case == 0:
            first = decoded.clone()
        if case == 3:
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
