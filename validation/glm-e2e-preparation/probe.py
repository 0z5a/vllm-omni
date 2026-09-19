"""Untimed GLM VAE and prior-token conditioning observations."""

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import TileTask
from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import GlmImagePipeline
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


def fingerprint(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }


class VaeProbeRunner(DiffusionModelRunner):
    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
        assert isinstance(self.pipeline, GlmImagePipeline)
        vae = self.pipeline.vae
        self.call = 0
        self.tile_calls = {"encode": 0, "decode": 0}
        self.path = Path(os.environ["GLM_PROBE_DIR"]) / f"probe-rank-{torch.distributed.get_rank()}.jsonl"
        self.vae_class = type(vae).__name__
        self.distributed = isinstance(vae, DistributedAutoencoderKL)
        if self.distributed:
            encode_tile, decode_tile = vae.encode_tile_exec, vae.tile_exec

            def record_encode_tile(task: TileTask) -> torch.Tensor:
                self.tile_calls["encode"] += 1
                return encode_tile(task)

            def record_decode_tile(task: TileTask) -> torch.Tensor:
                self.tile_calls["decode"] += 1
                return decode_tile(task)

            vae.encode_tile_exec = record_encode_tile
            vae.tile_exec = record_decode_tile
        original_encode, original_decode = vae.encode, vae.decode

        def encode(image: torch.Tensor, return_dict: bool = True):
            result = original_encode(image, return_dict=return_dict)
            posterior = result.latent_dist if return_dict else result[0]
            self.record("encode", image, posterior.parameters)
            return result

        def decode(latent: torch.Tensor, return_dict: bool = True, generator: torch.Generator | None = None):
            result = original_decode(latent, return_dict=return_dict, generator=generator)
            self.record("decode", latent, result.sample if return_dict else result[0])
            return result

        vae.encode, vae.decode = encode, decode
        self.pipeline.transformer.register_forward_pre_hook(self.record_prior, with_kwargs=True)

    def record_prior(self, module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        prior = cast(torch.Tensor, kwargs["prior_token_id"])
        embedding = cast(torch.Tensor, kwargs["encoder_hidden_states"])
        self.record("prior_and_text", prior, embedding)

    def record(self, kind: str, before: torch.Tensor, after: torch.Tensor) -> None:
        row = {
            "rank": torch.distributed.get_rank(),
            "call": self.call,
            "kind": kind,
            "vae_class": self.vae_class,
            "distributed": self.distributed,
            "tile_calls": dict(self.tile_calls),
            "input": fingerprint(before),
            "output": fingerprint(after),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.call += 1
