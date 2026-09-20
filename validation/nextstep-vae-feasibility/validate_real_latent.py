"""Compare native and two-rank halo decode on a captured NextStep AR latent."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.nextstep_1_1.modeling_flux_vae import AutoencoderKL

RAW_MEAN_LIMIT = 0.002
RAW_MAX_LIMIT = 0.02
RELATIVE_L2_LIMIT = 0.01
PIXEL_P99_LIMIT = 2


def synchronize_seconds(started: float) -> float:
    torch.accelerator.synchronize()
    elapsed = torch.tensor(time.perf_counter() - started, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    model = AutoencoderKL.from_pretrained(Path(args.model) / "vae", torch_dtype=torch.bfloat16).cuda().eval()
    latent = torch.load(args.latent, map_location="cpu", weights_only=True).to(device="cuda", dtype=torch.bfloat16)
    split = latent.shape[-1] // 2
    halo = max(
        int(model.tile_latent_min_size * model.tile_overlap_factor) // 2,
        min(latent.shape[-2], split) // 2,
    )
    local = latent[:, :, :, : split + halo] if rank == 0 else latent[:, :, :, split - halo :]
    native_times: list[float] = []
    candidate_times: list[float] = []
    native = torch.empty(0, device="cuda")
    candidate = torch.empty(0, device="cuda")
    with torch.inference_mode():
        for iteration in range(args.warmups + args.repeats):
            dist.barrier()
            if rank == 0:
                torch.accelerator.synchronize()
                started = time.perf_counter()
                native = model.decode(latent).sample
            native_seconds = synchronize_seconds(started if rank == 0 else time.perf_counter())
            if iteration >= args.warmups and rank == 0:
                native_times.append(native_seconds)

            dist.barrier()
            torch.accelerator.synchronize()
            started = time.perf_counter()
            decoded = model.decode(local).sample
            cropped = decoded[:, :, :, : split * 8] if rank == 0 else decoded[:, :, :, halo * 8 :]
            gathered = [torch.empty_like(cropped) for _ in range(2)]
            dist.all_gather(gathered, cropped)
            if rank == 0:
                candidate = torch.cat(gathered, dim=-1)
            candidate_seconds = synchronize_seconds(started)
            if iteration >= args.warmups and rank == 0:
                candidate_times.append(candidate_seconds)

    if rank == 0:
        assert native.shape == candidate.shape
        delta = (candidate.float() - native.float()).abs()
        relative_l2 = (delta.norm() / native.float().norm()).item()
        native_pixels = ((native.float() + 1) * 127.5).clamp(0, 255).round().to(torch.int16)
        candidate_pixels = ((candidate.float() + 1) * 127.5).clamp(0, 255).round().to(torch.int16)
        pixel_delta = (candidate_pixels - native_pixels).abs().float()
        raw_mean_abs = delta.mean().item()
        raw_max_abs = delta.max().item()
        pixel_p99_abs = torch.quantile(pixel_delta, 0.99).item()
        accepted = (
            raw_mean_abs <= RAW_MEAN_LIMIT
            and raw_max_abs <= RAW_MAX_LIMIT
            and relative_l2 <= RELATIVE_L2_LIMIT
            and pixel_p99_abs <= PIXEL_P99_LIMIT
        )
        metrics = {
            "latent_shape": list(latent.shape),
            "halo_latents": halo,
            "raw_mean_abs": raw_mean_abs,
            "raw_max_abs": raw_max_abs,
            "relative_l2": relative_l2,
            "pixel_p99_abs": pixel_p99_abs,
            "changed_pixel_fraction": (pixel_delta > 0).float().mean().item(),
            "native_median_seconds": statistics.median(native_times),
            "two_gpu_median_seconds": statistics.median(candidate_times),
            "speedup": statistics.median(native_times) / statistics.median(candidate_times),
            "native_seconds": native_times,
            "two_gpu_seconds": candidate_times,
            "native_sha256": hashlib.sha256(native.float().cpu().numpy().tobytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate.float().cpu().numpy().tobytes()).hexdigest(),
            "accepted": accepted,
        }
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
