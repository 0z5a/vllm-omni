"""Real-weight Qwen Edit VAE component parity; not full-pipeline validation."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(ulysses_degree=2)
    for slug in ("qwen-image-edit", "qwen-image-edit-2509"):
        model = Path.home() / "omni-1217-20260919/models" / slug
        native = (
            AutoencoderKLQwenImage.from_pretrained(model, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
        )
        candidate = (
            DistributedAutoencoderKLQwenImage.from_pretrained(model, subfolder="vae", torch_dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        native.enable_tiling()
        candidate.enable_tiling()
        candidate.set_parallel_size(2)
        torch.manual_seed(142)
        with torch.inference_mode():
            for batch, frames, height, width in ((1, 1, 320, 448), (2, 5, 448, 320), (1, 1, 64, 64), (1, 1, 320, 448)):
                pixels = torch.randn(batch, 3, frames, height, width, device="cuda", dtype=torch.bfloat16)
                before = torch.cuda.get_rng_state()
                expected = native.encode(pixels).latent_dist
                actual = candidate.encode(pixels).latent_dist
                assert torch.equal(torch.cuda.get_rng_state(), before)
                torch.testing.assert_close(actual.parameters, expected.parameters, rtol=0, atol=0)
                torch.testing.assert_close(actual.mode(), expected.mode(), rtol=0, atol=0)
                torch.testing.assert_close(
                    actual.sample(generator=torch.Generator(device="cuda").manual_seed(42)),
                    expected.sample(generator=torch.Generator(device="cuda").manual_seed(42)),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(candidate.tiled_encode(pixels), native.tiled_encode(pixels), rtol=0, atol=0)
                latent = expected.mode()
                reference_image = native.decode(latent).sample
                decoded = candidate.decode(latent).sample
                torch.testing.assert_close(decoded, reference_image, rtol=0, atol=0)
                print(
                    json.dumps(
                        {
                            "model": slug,
                            "rank": rank,
                            "shape": list(pixels.shape),
                            "encode_exact": True,
                            "decode_exact": True,
                            "rng_unchanged": True,
                            "forced_tiled_exact": True,
                        }
                    ),
                    flush=True,
                )
        del native, candidate, pixels, expected, actual, latent, reference_image, decoded
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
