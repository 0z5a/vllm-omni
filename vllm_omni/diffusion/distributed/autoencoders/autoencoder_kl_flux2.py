# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL_base
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLFlux2(DistributedAutoencoderKL_base, AutoencoderKLFlux2):
    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        if not self.is_distributed_enabled():
            kwargs["return_dict"] = return_dict
            return super().decode(z, *args, **kwargs)

        split, exec_fn, merge = self._strategy_select(z)

        if split is not None:
            strategy = "tile" if split == self.tile_split else "patch"
            logger.info(f"Decode run with distributed executor, split strategy is {strategy}")
            result = self.distributed_executor.execute(
                z,
                DistributedOperator(split=split, exec=exec_fn, merge=merge),
                broadcast_result=True,
            )
            if not return_dict:
                return (result,)

            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=result)

        kwargs["return_dict"] = return_dict
        return super().decode(z, *args, **kwargs)
