"""Measure encoder CUDA invocations only during the untimed native audit."""

import time

import torch

from harness.audit_ovis_worker_public import OvisPublicAudit


class OvisPublicLatencyAudit(OvisPublicAudit):
    def ovis_begin_audit(self):
        before = super().ovis_begin_audit()
        self._encoder_calls = []

        def begin(module, inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._encoder_call_started = (event, time.perf_counter())

        def end(module, inputs, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            start, host_start = self._encoder_call_started
            self._encoder_calls.append(
                (start, event, (time.perf_counter() - host_start) * 1000)
            )

        encoder = self.model_runner.pipeline.text_encoder
        self._encoder_latency_handles = [
            encoder.register_forward_pre_hook(begin),
            encoder.register_forward_hook(end),
        ]
        return before

    def ovis_end_audit(self):
        for handle in self._encoder_latency_handles:
            handle.remove()
        after = super().ovis_end_audit()
        calls = []
        for start, end, dispatch in self._encoder_calls:
            end.synchronize()
            calls.append(
                {"cuda_ms": start.elapsed_time(end), "host_dispatch_ms": dispatch}
            )
        after["encoder_calls"] = calls
        after["encoder_latency_scope"] = (
            "Untimed positive/negative encoder forward CUDA events; "
            "excludes tokenization and input preparation."
        )
        return after
