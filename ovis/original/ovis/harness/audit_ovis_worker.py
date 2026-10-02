"""Audit actual FSDP gathers and request cleanup outside timed Ovis requests."""

from collections import Counter
from types import MethodType

import torch
from torch.distributed.fsdp import FSDPModule

from harness.audit_worker import PipelineAudit


class OvisAudit(PipelineAudit):
    def pipeline_snapshot(self):
        snapshot = super().pipeline_snapshot()
        snapshot["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        snapshot["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        snapshot["hsdp_groups"] = self.ovis_shard_snapshot()
        return snapshot

    def ovis_shard_snapshot(self):
        pipeline = self.model_runner.pipeline
        return [
            {
                "module": name,
                "group": index,
                "sharded": group.is_sharded,
                "reshard_after_forward": group._reshard_after_forward,
            }
            for name, module in pipeline.transformer.named_modules()
            if isinstance(module, FSDPModule)
            for index, group in enumerate(module._get_fsdp_state()._fsdp_param_groups)
        ]

    def ovis_begin_audit(self):
        pipeline = self.model_runner.pipeline
        self._ovis_gathers = Counter()
        self._ovis_originals = []
        self._ovis_decode_state = []
        for name, module in pipeline.transformer.named_modules():
            if not isinstance(module, FSDPModule):
                continue
            for index, group in enumerate(module._get_fsdp_state()._fsdp_param_groups):
                original = group.unshard
                label = f"{name}:{index}"

                def unshard(group, async_op=False, *, original=original, label=label):
                    pending = group._all_gather_result is not None
                    original(async_op)
                    if not pending and group._all_gather_result is not None:
                        self._ovis_gathers[label] += 1

                self._ovis_originals.append((group, original))
                group.unshard = MethodType(unshard, group)
        self._ovis_original_decode = pipeline.vae.decode

        def decode(*args, **kwargs):
            state = self.ovis_shard_snapshot()
            self._ovis_decode_state.append(state)
            if (
                pipeline.od_config.additional_config.get(
                    "hsdp_reshard_after_forward", True
                )
                is False
            ):
                assert state and all(
                    row["sharded"] and row["reshard_after_forward"] for row in state
                )
            return self._ovis_original_decode(*args, **kwargs)

        pipeline.vae.decode = decode
        return {"rank": self.rank, "before": self.ovis_shard_snapshot()}

    def ovis_end_audit(self):
        pipeline = self.model_runner.pipeline
        for group, original in self._ovis_originals:
            group.unshard = original
        pipeline.vae.decode = self._ovis_original_decode
        state = self.ovis_shard_snapshot()
        resident = (
            pipeline.od_config.additional_config.get("hsdp_reshard_after_forward", True)
            is False
        )
        if resident:
            assert state and all(
                row["sharded"] and row["reshard_after_forward"] for row in state
            )
            assert self._ovis_gathers and all(
                count == 1 for count in self._ovis_gathers.values()
            )
        return {
            "rank": self.rank,
            "gathers": dict(self._ovis_gathers),
            "before_vae": self._ovis_decode_state,
            "after": state,
            "resident": resident,
        }

    def ovis_exception_cleanup(self):
        pipeline = self.model_runner.pipeline
        assert isinstance(pipeline.transformer, FSDPModule)
        assert (
            pipeline.od_config.additional_config["hsdp_reshard_after_forward"] is False
        )
        try:
            with pipeline._hsdp_denoising_context():
                for module in pipeline.transformer.modules():
                    if isinstance(module, FSDPModule):
                        module.unshard(async_op=False)
                raise RuntimeError("requested residency cleanup check")
        except RuntimeError as error:
            assert str(error) == "requested residency cleanup check"
        state = self.ovis_shard_snapshot()
        assert state and all(
            row["sharded"] and row["reshard_after_forward"] for row in state
        )
        return {"rank": self.rank, "after_injected_exception": state}

    def audit_stop_profile(self, path):
        super().audit_stop_profile(path.replace(".json", f"-rank{self.rank}.json"))
