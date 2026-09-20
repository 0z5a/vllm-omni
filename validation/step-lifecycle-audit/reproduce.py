"""Inspect real runner ownership after cancellation/completion using its CPU test pipeline."""

import json
import weakref

import pytest

from tests.diffusion.test_diffusion_step_pipeline import (
    _make_cached_scheduler_output,
    _make_runner,
    _make_scheduler_output,
    _make_step_request,
    _noop_forward_context,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module


def main() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "set_forward_context", _noop_forward_context)
        for outcome in ("cancel", "complete"):
            runner = _make_runner()
            request = _make_step_request()
            runner.execute_stepwise(_make_scheduler_output(request))
            state = weakref.ref(runner.state_cache[request.request_id])
            if outcome == "complete":
                result = runner.execute_stepwise(_make_cached_scheduler_output())
                assert result.get_request_output(request.request_id).finished
            cleanup = DiffusionSchedulerOutput(
                step_id=2,
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                finished_req_ids={request.request_id},
                num_running_reqs=0,
                num_waiting_reqs=0,
            )
            runner.execute_stepwise(cleanup)
            print(
                json.dumps(
                    {
                        "outcome": outcome,
                        "state_cache_empty": not runner.state_cache,
                        "batch_request_ids": runner.input_batch.request_ids if runner.input_batch is not None else [],
                        "batch_state_ids": [item.request_id for item in runner.input_batch.states]
                        if runner.input_batch is not None
                        else [],
                        "request_state_still_owned": state() is not None,
                        "scope": "CPU test pipeline through real runner; no GPU memory or full-model claim",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
