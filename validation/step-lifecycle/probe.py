"""Untimed Helios lifecycle observations through the existing worker RPC API."""

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend, SequentialOffloadHook
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, StepRequestState
from vllm_omni.lora.request import LoRARequest


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class LifecycleProbe:
    model_runner: DiffusionModelRunner
    rank: int
    _step_lora_state: dict[str, tuple[LoRARequest | None, float]]

    def lifecycle_install(self) -> None:
        pipeline = self.model_runner.pipeline
        assert isinstance(pipeline, HeliosPipeline)
        self.lifecycle_events: list[dict[str, object]] = []
        self.lifecycle_moves: list[dict[str, object]] = []
        self.lifecycle_waves: list[dict[str, object]] = []
        self.lifecycle_failure: str | None = None
        original_prepare = pipeline.prepare_encode

        def prepare(state: StepRequestState) -> StepRequestState:
            if state.request_id == self.lifecycle_failure:
                self.lifecycle_failure = None
                self.lifecycle_events.append({"component": "injected_failure", "request": state.request_id})
                raise ValueError("injected rank-local preparation failure")
            return original_prepare(state)

        pipeline.prepare_encode = prepare
        original_step = self.model_runner.execute_stepwise

        def step(schedule: DiffusionSchedulerOutput) -> BatchRunnerOutput:
            result = original_step(schedule)
            self.lifecycle_waves.append(
                {
                    "step_id": schedule.step_id,
                    "scheduled": schedule.scheduled_request_ids,
                    "outputs": [[out.request_id, out.step_index, out.finished] for out in result.runner_outputs],
                }
            )
            return result

        self.model_runner.execute_stepwise = step
        original_move = SequentialOffloadHook._move_params

        def move(
            module: nn.Module,
            target_device: torch.device,
            *,
            non_blocking: bool = False,
            pin_memory: bool = False,
        ) -> bool:
            changed = original_move(module, target_device, non_blocking=non_blocking, pin_memory=pin_memory)
            if changed:
                self.lifecycle_moves.append(
                    {"module": type(module).__name__, "target": str(target_device), "wave": len(self.lifecycle_waves)}
                )
            return changed

        SequentialOffloadHook._move_params = staticmethod(move)
        for name, module in (
            ("dit", pipeline.transformer.blocks[0]),
            ("text_encoder", pipeline.text_encoder.encoder.block[0]),
            ("vae", pipeline.vae.decoder),
        ):

            def observe(component: nn.Module, inputs: tuple, label: str = name) -> None:
                parameter = next(component.parameters())
                assert parameter.device.type == "cuda", (label, parameter.device)
                self.lifecycle_events.append({"component": label, "device": str(parameter.device)})

            module.register_forward_pre_hook(observe)

    def lifecycle_fail_next(self, request_id: str, failed_rank: int) -> None:
        self.lifecycle_failure = request_id if self.rank == failed_rank else None

    def lifecycle_snapshot(self, label: str) -> None:
        runner = self.model_runner
        pipeline = runner.pipeline
        assert isinstance(pipeline, HeliosPipeline)
        states = {}
        for request_id, state in runner.state_cache.items():
            generator = state.sampling.generator
            assert isinstance(generator, torch.Generator)
            states[request_id] = {
                "step": state.step_index,
                "chunk": state.chunk_index,
                "rng": tensor_hash(generator.get_state()),
                "latents": tensor_hash(state.latents) if state.latents is not None else None,
                "prompt": tensor_hash(state.prompt_embeds) if state.prompt_embeds is not None else None,
                "history": tensor_hash(state.extra["history_latents"]),
            }
        components = {}
        for name, module in (
            ("dit", pipeline.transformer),
            ("text_encoder", pipeline.text_encoder),
            ("vae", pipeline.vae),
        ):
            devices: dict[str, int] = {}
            sharded = 0
            for parameter in module.parameters():
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                devices[str(local.device)] = devices.get(str(local.device), 0) + local.numel() * local.element_size()
                sharded += isinstance(parameter, DTensor)
            components[name] = {"bytes_by_device": devices, "dtensors": sharded}
        source_root = Path(__file__).resolve().parent
        import vllm_omni.diffusion.worker.diffusion_model_runner as runner_module

        row = {
            "label": label,
            "rank": self.rank,
            "pid": os.getpid(),
            "states": states,
            "batch_ids": None if runner.input_batch is None else runner.input_batch.request_ids,
            "lora_ids": sorted(self._step_lora_state),
            "components": components,
            "fsdp_modules": sum(isinstance(module, FSDPModule) for module in pipeline.transformer.modules()),
            "events": self.lifecycle_events,
            "moves": self.lifecycle_moves,
            "waves": self.lifecycle_waves,
            "module_offload_enabled": isinstance(runner.offload_backend, ModelLevelOffloadBackend)
            and runner.offload_backend.enabled,
            "runner_file": runner_module.__file__,
            "runner_sha256": hashlib.sha256(Path(runner_module.__file__).read_bytes()).hexdigest(),
            "probe_sha256": hashlib.sha256((source_root / "probe.py").read_bytes()).hexdigest(),
        }
        output = Path(os.environ["STEP_LIFECYCLE_OUTPUT"])
        with (output / f"rank-{self.rank}.jsonl").open("a") as records:
            records.write(json.dumps(row) + "\n")
