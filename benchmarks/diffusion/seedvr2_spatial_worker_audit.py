# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record the active SeedVR2 spatial group during untimed worker warmup."""

import json
import os
from pathlib import Path

import torch.distributed as dist
from seedvr2_native_worker_audit import NativeWorkerAudit


class SpatialWorkerAudit(NativeWorkerAudit):
    def _write_seedvr2_audit(self):
        super()._write_seedvr2_audit()
        vae = self.model_runner.pipeline.vae
        group = vae._spatial_group
        path = Path(os.environ["SEEDVR2_AUDIT_DIR"]) / f"worker-{os.getpid()}.json"
        record = json.loads(path.read_text())
        record["vae_execution"] = {
            "use_tiling": vae.use_tiling,
            "spatial_world_size": 1 if group is None else dist.get_world_size(group),
            "spatial_rank": 0 if group is None else dist.get_rank(group),
        }
        path.write_text(json.dumps(record, indent=2))
