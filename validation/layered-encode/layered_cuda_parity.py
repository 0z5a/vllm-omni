"""Real-weight two-rank Layered VAE parity; this is not full-model E2E."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage_layered import (
    DistributedAutoencoderKLQwenImageLayered,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(ulysses_degree=2)
    model = Path.home() / "omni-1217-20260919/models/qwen-image-layered"
    native = (
        AutoencoderKLQwenImage.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    candidate = (
        DistributedAutoencoderKLQwenImageLayered.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    native.enable_tiling()
    candidate.enable_tiling()
    candidate.set_parallel_size(2)
    torch.manual_seed(142)
    with torch.inference_mode():
        for batch, frames, height, width in (
            (1, 1, 320, 448),
            (2, 5, 448, 320),
            (1, 1, 320, 448),
        ):
            pixels = torch.randn(
                batch,
                native.config.input_channels,
                frames,
                height,
                width,
                device="cuda",
                dtype=torch.bfloat16,
            )
            rng_before = torch.cuda.get_rng_state()
            expected = native.encode(pixels).latent_dist
            actual = candidate.encode(pixels).latent_dist
            assert torch.equal(torch.cuda.get_rng_state(), rng_before)
            torch.testing.assert_close(
                actual.parameters, expected.parameters, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.sample(generator=torch.Generator(device="cuda").manual_seed(42)),
                expected.sample(
                    generator=torch.Generator(device="cuda").manual_seed(42)
                ),
                rtol=0,
                atol=0,
            )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "shape": list(pixels.shape),
                        "exact": True,
                        "rng_unchanged": True,
                    }
                ),
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
