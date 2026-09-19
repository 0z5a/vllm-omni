"""Real-checkpoint CPU API audit; does not claim distributed NextStep VAE support."""

import hashlib
import json
from pathlib import Path

import torch

from vllm_omni.diffusion.models.nextstep_1_1.modeling_flux_vae import AutoencoderKL


def main() -> None:
    root = Path.home() / "omni-1217-20260919/models/nextstep/vae"
    model = AutoencoderKL.from_pretrained(root).eval()
    state = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert set(state) == set(model.state_dict())
    del state
    scale = 2 ** (len(model.config.block_out_channels) - 1)
    with torch.inference_mode():
        for batch, height, width in ((1, 16, 24), (2, 32, 48)):
            torch.manual_seed(142)
            latent = torch.randn(batch, model.config.latent_channels, height, width)
            model.use_tiling = False
            native = model.decode(latent).sample
            model.use_tiling = True
            flagged = model.decode(latent, return_dict=False)[0]
            torch.testing.assert_close(flagged, native, rtol=0, atol=0)
            assert native.shape == (batch, 3, height * scale, width * scale)
            split = width // 2
            halo = max(int(model.tile_latent_min_size * model.tile_overlap_factor) // 2, min(height, split) // 2)
            left = model.decode(latent[:, :, :, : split + halo]).sample[:, :, :, : split * scale]
            right = model.decode(latent[:, :, :, split - halo :]).sample[:, :, :, halo * scale :]
            patched = torch.cat((left, right), dim=-1)
            assert patched.shape == native.shape and torch.isfinite(patched).all()
            delta = (patched - native).abs()
            print(
                json.dumps(
                    {
                        "latent_shape": list(latent.shape),
                        "output_shape": list(native.shape),
                        "dtype": str(native.dtype),
                        "tile_flag_changes_output": False,
                        "checkpoint_keys_match": True,
                        "halo_latents": halo,
                        "halo_max_abs": delta.max().item(),
                        "halo_mean_abs": delta.mean().item(),
                        "halo_relative_l2": (delta.norm() / native.norm()).item(),
                        "native_sha256": hashlib.sha256(native.numpy().tobytes()).hexdigest(),
                        "claim": "CPU API audit; naive patch decoding is an approximation, not native tiling",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
