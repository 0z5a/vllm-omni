"""Inspect the loaded production pipeline and record an untimed CUDA trace."""

import hashlib
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm_omni.diffusion.models.flux.pipeline_flux_kontext import FluxKontextPipeline
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline
from vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image import OvisImagePipeline


class PipelineAudit:
    def audit_disable_encoder_graph(self):
        pipeline = self.model_runner.pipeline
        assert isinstance(pipeline, Flux2KleinPipeline)
        graph = pipeline._text_encoder_graph
        assert graph is not None
        captured = graph.graph is not None
        if graph._replay_done is not None:
            graph._replay_done.synchronize()
        pipeline._text_encoder_graph = None
        return {
            "native_graph_eligible": True,
            "captured_before_control": captured,
            "encoder_graph_disabled_for_comparison": True,
        }

    def pipeline_snapshot(self):
        pipeline = self.model_runner.pipeline
        assert isinstance(pipeline, (FluxKontextPipeline, Flux2KleinPipeline, OvisImagePipeline))
        projections = []
        for name, layer in pipeline.named_modules():
            if isinstance(layer, LinearBase):
                method = layer.quant_method
                kernel = vars(method).get("fp8_linear")
                projections.append(
                    {
                        "name": name,
                        "method": type(method).__name__,
                        "backend": type(kernel).__name__ if kernel is not None else None,
                        "dtype": str(layer.weight.dtype),
                        "shape": list(layer.weight.shape),
                        "device": str(layer.weight.device),
                    }
                )
        graph_state = None
        if isinstance(pipeline, Flux2KleinPipeline):
            graph = pipeline._text_encoder_graph
            graph_state = {"eligible": graph is not None, "captured": graph is not None and graph.graph is not None}
        sources = {}
        for name in [
            type(pipeline).__module__,
            "vllm.model_executor.layers.quantization.fp8",
            "vllm_omni.diffusion.executor.uniproc_executor",
        ]:
            module = importlib.import_module(name)
            path = Path(module.__file__)
            sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return {
            "pid": os.getpid(),
            "rank": self.rank,
            "python": sys.executable,
            "pipeline": type(pipeline).__name__,
            "gpu": torch.cuda.get_device_name(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "vllm",
                    "transformers",
                    "diffusers",
                    "huggingface-hub",
                    "accelerate",
                    "triton",
                    "flashinfer-python",
                ]
            },
            "sources": sources,
            "projections": projections,
            "text_encoder_graph": graph_state,
            "torch_compile": {
                "enforce_eager": self.model_runner.od_config.enforce_eager,
                "granularity": self.model_runner.od_config.diffusion_compile_granularity,
                "compiled_forwards": [
                    name
                    for name, module in pipeline.named_modules()
                    if "_torchdynamo_orig_callable" in vars(module.forward)
                ],
                "stats": dict(torch._dynamo.utils.counters["stats"]),
                "inductor": dict(torch._dynamo.utils.counters["inductor"]),
            },
            "vae_dtypes": sorted({str(p.dtype) for p in pipeline.vae.parameters()}),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }

    def audit_start_profile(self):
        self._audit_profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        self._audit_profile.__enter__()

    def audit_stop_profile(self, path):
        self._audit_profile.__exit__(None, None, None)
        self._audit_profile.export_chrome_trace(path)
        self._audit_profile = None
