"""Record native encoder weights and untimed activation dtypes."""

from collections import Counter

import torch
from vllm.model_executor.layers.linear import LinearBase

from harness.audit_ovis_worker import OvisAudit


class OvisPublicAudit(OvisAudit):
    def pipeline_snapshot(self):
        snapshot = super().pipeline_snapshot()
        encoder = self.model_runner.pipeline.text_encoder
        snapshot["encoder_parameter_dtypes"] = dict(
            Counter(str(parameter.dtype) for parameter in encoder.parameters())
        )
        snapshot["encoder_native_linears"] = [
            {
                "name": name,
                "dtype": str(module.weight.dtype),
                "shape": list(module.weight.shape),
                "device": str(module.weight.device),
            }
            for name, module in encoder.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        return snapshot

    def ovis_begin_audit(self):
        before = super().ovis_begin_audit()
        self._encoder_activation_dtypes = Counter()

        def record_dtype(module, inputs):
            self._encoder_activation_dtypes[str(inputs[0].dtype)] += 1

        self._encoder_dtype_handles = [
            module.register_forward_pre_hook(record_dtype)
            for module in self.model_runner.pipeline.text_encoder.modules()
            if isinstance(module, (torch.nn.Linear, LinearBase))
        ]
        return before

    def ovis_end_audit(self):
        for handle in self._encoder_dtype_handles:
            handle.remove()
        after = super().ovis_end_audit()
        after["encoder_activation_dtypes"] = dict(self._encoder_activation_dtypes)
        return after
