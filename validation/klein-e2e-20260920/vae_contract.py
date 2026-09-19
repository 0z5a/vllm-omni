"""Real Flux2 VAE contract and isolated component timing on WORLD=2."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import DistributedAutoencoderKLFlux2
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedOperator
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel


def error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - reference.float()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / reference.float().norm()).item(),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank)
    initialize_model_parallel(ulysses_degree=2)
    args.output.mkdir(parents=True, exist_ok=True)
    native = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    candidate = (
        DistributedAutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    candidate.enable_tiling()
    candidate.set_parallel_size(2)
    executor = candidate.distributed_executor
    assert dist.get_process_group_ranks(executor.group) == [0, 1]
    print(
        json.dumps(
            {
                "rank": rank,
                "group": [0, 1],
                "class": type(candidate).__name__,
                "load_peak_bytes": torch.accelerator.max_memory_allocated(),
                "weight_residency_bytes": torch.cuda.memory_allocated(),
                "tile_latent_min_size": native.tile_latent_min_size,
            }
        ),
        flush=True,
    )
    # Small single tile, halo patch, ragged 3x3 grid, batch >1, and reentry.
    cases = ((1, 16, 16), (1, 64, 64), (1, 128, 144), (2, 80, 48), (1, 16, 16))
    first_output = None
    with (args.output / f"rank-{rank}.jsonl").open("w") as records:
        for case, (batch, height, width) in enumerate(cases):
            generator = torch.Generator(device="cuda").manual_seed(142)
            latent = torch.randn(
                batch,
                native.config.latent_channels,
                height,
                width,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            if rank == 0:
                torch.save(latent.cpu(), args.output / f"latent-{case}.pt")
            tasks, grid = candidate.tile_split(latent)
            assigned = executor._balance_tasks(tasks, 2)
            outputs = {}
            modes = ["full", "native_tile", "distributed_tile", "distributed_tile_degree1"]
            if max(height, width) <= native.tile_latent_min_size:
                modes.append("halo_patch")
            for iteration in range(7):
                for mode in modes if iteration % 2 == 0 else reversed(modes):
                    dist.barrier()
                    torch.accelerator.synchronize()
                    torch.accelerator.reset_peak_memory_stats()
                    started = time.perf_counter()
                    actual = None
                    if mode.startswith("distributed_tile"):
                        candidate.set_parallel_size(1 if mode.endswith("degree1") else 2)
                        actual = executor.execute(
                            latent, DistributedOperator(candidate.tile_split, candidate.tile_exec, candidate.tile_merge)
                        )
                    elif mode == "halo_patch":
                        candidate.set_parallel_size(2)
                        actual = candidate.decode(latent, return_dict=False)[0]
                    elif rank == 0:
                        native.disable_tiling()
                        actual = (
                            native.decode(latent, return_dict=False)[0]
                            if mode == "full"
                            else native.tiled_decode(latent, return_dict=False)[0]
                        )
                    torch.accelerator.synchronize()
                    seconds = time.perf_counter() - started
                    peak = torch.accelerator.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    elapsed = torch.tensor(seconds, device="cuda")
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    if actual is not None:
                        assert actual.dtype == latent.dtype and torch.isfinite(actual).all()
                        assert actual.shape == (batch, 3, height * 8, width * 8)
                        value = actual.cpu()
                        if mode in outputs:
                            assert torch.equal(value, outputs[mode]), (case, mode, "repeat drift")
                        outputs[mode] = value
                    row = {
                        "rank": rank,
                        "case": case,
                        "latent_shape": list(latent.shape),
                        "mode": mode,
                        "iteration": iteration,
                        "warmup": iteration < 2,
                        "seconds": elapsed.item(),
                        "peak_allocated": peak,
                        "peak_reserved": reserved,
                        "local_tiles": (len(tasks) if rank == 0 else 0)
                        if mode.endswith("degree1")
                        else len(assigned[rank]),
                        "grid": grid.grid_shape,
                    }
                    records.write(json.dumps(row) + "\n")
                    records.flush()
            if rank == 0:
                assert torch.equal(outputs["distributed_tile"], outputs["native_tile"])
                assert torch.equal(outputs["distributed_tile_degree1"], outputs["native_tile"])
                if case == 0:
                    first_output = outputs["distributed_tile"]
                if case == 4:
                    assert torch.equal(first_output, outputs["distributed_tile"])
                quality = {name: error(value, outputs["full"]) for name, value in outputs.items()}
                (args.output / f"quality-{case}.json").write_text(json.dumps(quality, indent=2) + "\n")
                print(json.dumps({"case": case, "quality_vs_full": quality, "tiled_exact": True}), flush=True)
            del latent, outputs, actual
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
