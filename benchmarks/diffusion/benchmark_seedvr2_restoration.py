# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline SeedVR2 E2E benchmark using the pinned reference VAE and sampler.

Requires a SeedVR reference checkout, its optional dependencies, and released
3B FP16 / VAE checkpoints. Measures video read through MP4 write, excluding load.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "models", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference))
    from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(0)
    args.output.mkdir(parents=True, exist_ok=False)
    config = OmegaConf.to_container(OmegaConf.load(args.reference / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    del config["__object__"]
    vae = VideoAutoencoderKLWrapper(**config, freeze_encoder=False)
    vae.load_state_dict(load_file(str(args.models / "ema_vae_fp16.safetensors")), strict=True)
    vae = vae.to(device=device, dtype=torch.float16).eval()
    vae.disable_slicing()
    vae.disable_tiling()
    model = SeedVR2NaDiT(**SEEDVR2_3B_CONFIG, use_varlen_kernel=False)
    weights = load_file(str(args.models / "seedvr2_ema_3b_fp16.safetensors"))
    normalized = {
        key.removesuffix(".rope.rope.freqs") + ".rope.freqs" if key.endswith(".rope.rope.freqs") else key: value
        for key, value in weights.items()
    }
    if len(normalized) != len(weights):
        raise ValueError("checkpoint key collision")
    model.load_state_dict(normalized, strict=True)
    del weights, normalized
    model = model.to(device=device, dtype=torch.float16).eval()
    text = torch.load(args.reference / "pos_emb.pt", map_location=device, weights_only=True).to(torch.float16)
    reference = VideoDiffusionInfer(
        OmegaConf.create(
            {
                "vae": {"dtype": "float16", "scaling_factor": 0.9152, "grouping": False},
                "diffusion": {
                    "schedule": {"type": "lerp", "T": 1000.0},
                    "sampler": {"type": "euler", "prediction_type": "v_lerp"},
                    "timesteps": {"sampling": {"type": "uniform_trailing", "steps": 1}},
                    "cfg": {"scale": 1.0, "rescale": 0.0},
                },
            }
        )
    )
    reference.vae = vae
    reference.dit = model
    reference.configure_diffusion()

    @torch.inference_mode()
    def request(index: int) -> dict[str, torch.Tensor]:
        with av.open(str(args.input)) as container:
            fps = container.streams.video[0].average_rate
            frames = []
            for frame in container.decode(video=0):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
                if len(frames) == args.frames:
                    break
        if len(frames) != args.frames:
            raise ValueError("input is shorter than the requested clip")
        sample = torch.stack(frames).permute(0, 3, 1, 2).float() / 255
        sample = resize(
            sample, [args.height, args.width], interpolation=InterpolationMode.BICUBIC, antialias=True
        ).clamp(0, 1)
        padding = (1 - len(sample)) % 4
        if padding:
            sample = torch.cat([sample, sample[-1:].repeat(padding, 1, 1, 1)])
        sample = (sample * 2 - 1).permute(1, 0, 2, 3)
        # This CLI owns its process and fixes the single visible device's seed.
        torch.manual_seed(7723)
        condition = reference.vae_encode([sample])[0]
        noise = torch.randn_like(condition)
        cond = reference.get_condition(noise, condition, "sr")
        velocity = model(
            vid=torch.cat([noise, cond], -1).reshape(-1, 33),
            txt=text,
            vid_shape=torch.tensor([condition.shape[:3]], device=device),
            txt_shape=torch.tensor([[len(text)]], device=device),
            timestep=torch.tensor([1000.0], device=device),
        ).vid_sample
        restored = reference.sampler.step_to(
            velocity, noise.reshape(-1, 16), torch.tensor(1000.0, device=device), torch.tensor(0.0, device=device)
        ).reshape_as(noise)
        decoded = reference.vae_decode([restored])[0]
        if decoded.ndim == 3:
            decoded = decoded.unsqueeze(1)
        output = ((decoded.permute(1, 0, 2, 3)[: args.frames].float() + 1) / 2).clamp(0, 1)
        rgb = (output.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        with av.open(str(args.output / f"{index}.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
            for pixels in rgb:
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return {"condition": condition, "velocity": velocity, "restored_latent": restored, "frames": output}

    timings = []
    for index in range(args.repeats + 1):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        result = request(index)
        torch.accelerator.synchronize()
        timings.append({"warmup": index == 0, "seconds": time.perf_counter() - start})
        if index == 0:
            torch.save({key: value.cpu() for key, value in result.items()}, args.output / "stages.pt")
        del result
    (args.output / "timing.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
