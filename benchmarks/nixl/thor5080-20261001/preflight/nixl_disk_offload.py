"""Keep native pageable AR offload weights in task-owned files on D:."""

import itertools
import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn
from vllm.model_executor.offloader import uva
from vllm.model_executor.model_loader import utils as loader_utils

ROOT = Path(os.environ["NIXL_TASK_ROOT"]) / "mapped-weights" / str(os.getpid())
COUNTER = itertools.count()
DEVICE_LOADING_CONTEXT = loader_utils.device_loading_context
BACKINGS: dict[int, tuple[Path, torch.Tensor]] = {}


def disk_backed(tensor: torch.Tensor, path: Path | None = None) -> torch.Tensor:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = path or ROOT / f"{next(COUNTER)}.bin"
    storage_size = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))
    storage = torch.from_file(
        str(path), shared=True,
        size=storage_size, dtype=tensor.dtype, device="cpu",
    )
    result = storage.as_strided(tensor.shape, tensor.stride())
    result.copy_(tensor)
    BACKINGS[result.data_ptr()] = (path, result)
    return result


class DiskUVAOffloader(uva.UVAOffloader):
    def _maybe_offload_to_cpu(self, module: nn.Module, prefix: str = "") -> nn.Module:
        assert not self.pin_memory and not self.uva_offloading
        candidates = {id(parameter) for parameter in module.parameters() if parameter.device.type == "cuda"}
        module = super()._maybe_offload_to_cpu(module, prefix)
        for parameter in module.parameters():
            if id(parameter) in candidates and parameter.device.type == "cpu" and parameter.numel() >= 65536:
                parameter.data = disk_backed(parameter.data)
        return module


def install() -> None:
    uva.UVAOffloader = DiskUVAOffloader
    loader_utils.device_loading_context = disk_loading_context


@contextmanager
def disk_loading_context(module: nn.Module, target_device: torch.device) -> Iterator[nn.Module]:
    if target_device.type == "cpu":
        with DEVICE_LOADING_CONTEXT(module, target_device) as loaded:
            yield loaded
        return
    paths = {}
    for name, parameter in module.named_parameters():
        backing = BACKINGS.pop(parameter.data_ptr(), None)
        if backing is not None:
            paths[name] = backing[0]
    with DEVICE_LOADING_CONTEXT(module, target_device) as loaded:
        yield loaded
    for name, parameter in module.named_parameters():
        if parameter.device.type == "cpu" and parameter.numel() >= 65536:
            parameter.data = disk_backed(parameter.data, paths.get(name))
