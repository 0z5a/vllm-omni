# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark source audit at worker startup and through the five request warmups."""

import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import torch
from vllm.triton_utils import triton


class NativeWorkerAudit:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seedvr2_audit_rank = kwargs["rank"]
        self._seedvr2_audit_forwards = 0
        self._write_seedvr2_audit()
        pipeline = self.model_runner.pipeline
        forward = pipeline.forward

        def warmup_forward(batch):
            output = forward(batch)
            self._seedvr2_audit_forwards += 1
            if self._seedvr2_audit_forwards == 6:
                pipeline.forward = forward
            self._write_seedvr2_audit()
            return output

        pipeline.forward = warmup_forward

    def _write_seedvr2_audit(self):
        modules = {}
        for name, module in tuple(sys.modules.items()):
            if name.startswith("vllm_omni.diffusion.models.seedvr2"):
                path = Path(module.__file__)
                modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        kernels = {}
        name = "vllm_omni.diffusion.models.seedvr2.frame_norm"
        if name in sys.modules:
            from vllm_omni.diffusion.models.seedvr2 import frame_norm

            for function in (frame_norm._moments, frame_norm._statistics, frame_norm._normalize):
                for device_cache in function.device_caches.values():
                    for kernel in device_cache[0].values():
                        artifacts = {}
                        for filename, path in kernel.metadata_group.items():
                            if filename.endswith((".cubin", ".json")):
                                artifacts[filename] = {
                                    "path": path,
                                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                }
                        kernels[kernel.hash] = {
                            "name": kernel.name,
                            "source_key": function.hash,
                            "artifacts": artifacts,
                        }
        record = {
            "pid": os.getpid(),
            "time": time.time(),
            "python": sys.executable,
            "modules": modules,
            "triton_version": triton.__version__,
            "compiled_kernels": kernels,
            "rank": self._seedvr2_audit_rank,
            "audited_forwards": self._seedvr2_audit_forwards,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "diffusers", "av")},
            "loaded_allocated": torch.accelerator.memory_allocated(),
            "loaded_reserved": torch.accelerator.memory_reserved(),
        }
        path = Path(os.environ["SEEDVR2_AUDIT_DIR"]) / f"worker-{os.getpid()}.json"
        path.write_text(json.dumps(record, indent=2))
