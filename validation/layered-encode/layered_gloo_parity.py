"""Two-process CPU collective check; no full-model or GPU performance claim."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage_layered import DistributedAutoencoderKLQwenImageLayered
from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import AutoencoderKLQwenImage


def main():
    dist.init_process_group('gloo')
    torch.manual_seed(142)
    native = AutoencoderKLQwenImage(base_dim=4, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1).eval()
    candidate = DistributedAutoencoderKLQwenImageLayered.from_config(native.config).eval()
    candidate.load_state_dict(native.state_dict())
    with patch('vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor.get_world_group',
               return_value=SimpleNamespace(device_group=dist.group.WORLD)):
        candidate.init_distributed()
    candidate.set_parallel_size(2)
    for model in (native, candidate):
        model.enable_tiling(tile_sample_min_height=16, tile_sample_min_width=16,
                            tile_sample_stride_height=8, tile_sample_stride_width=8)
    with torch.inference_mode():
        for shape in ((1, 3, 1, 24, 32), (2, 3, 5, 32, 24), (1, 3, 1, 24, 32)):
            x = torch.randn(shape)
            rng = torch.random.get_rng_state()
            expected = native.encode(x).latent_dist
            actual = candidate.encode(x).latent_dist
            torch.testing.assert_close(actual.parameters, expected.parameters, rtol=0, atol=0)
            assert torch.equal(torch.random.get_rng_state(), rng)
            torch.testing.assert_close(actual.sample(generator=torch.Generator().manual_seed(42)),
                                       expected.sample(generator=torch.Generator().manual_seed(42)), rtol=0, atol=0)
            print(json.dumps(dict(rank=dist.get_rank(),shape=shape,exact=True,rng_unchanged=True)),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
