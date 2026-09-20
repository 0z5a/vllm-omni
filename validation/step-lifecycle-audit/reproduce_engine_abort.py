"""Reproduce last-request abort ownership through the real engine and CPU runner."""

import json
from types import SimpleNamespace

import pytest

from tests.diffusion.test_diffusion_step_pipeline import (
    _make_engine,
    _make_engine_request,
    _make_runner,
    _noop_forward_context,
)
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput


def main() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "set_forward_context", _noop_forward_context)
        runner = _make_runner()
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("abort-last", num_inference_steps=4)

        def execute(schedule: DiffusionSchedulerOutput) -> BatchRunnerOutput:
            output = runner.execute_stepwise(schedule)
            engine.abort(request.request_id)
            return output

        engine.execute_fn = execute
        output = engine.add_req_and_wait_for_response(request)
        assert output.aborted
        print(
            json.dumps(
                {
                    "engine_returned_aborted": output.aborted,
                    "scheduler_has_requests": scheduler.has_requests(),
                    "worker_cached_requests": list(runner.state_cache),
                    "worker_batch_requests": runner.input_batch.request_ids if runner.input_batch is not None else [],
                    "scope": "Synchronous engine and runner with CPU test pipeline; no GPU/asynchronous-service claim",
                }
            )
        )


if __name__ == "__main__":
    main()
