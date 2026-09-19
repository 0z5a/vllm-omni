# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup-only benchmark audit via the framework worker extension hook."""

import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import torch


class NativeWorkerAudit:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = {}
        for name, module in tuple(sys.modules.items()):
            if name.startswith("vllm_omni.diffusion.models.seedvr2"):
                path = Path(module.__file__)
                modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        record = {
            "pid": os.getpid(),
            "time": time.time(),
            "python": sys.executable,
            "modules": modules,
            "rank": kwargs["rank"],
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "diffusers", "av")},
            "loaded_allocated": torch.accelerator.memory_allocated(),
            "loaded_reserved": torch.accelerator.memory_reserved(),
        }
        path = Path(os.environ["SEEDVR2_AUDIT_DIR"]) / f"worker-{os.getpid()}.json"
        path.write_text(json.dumps(record, indent=2))
