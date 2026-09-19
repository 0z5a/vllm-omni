# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("inference_mode", [False, True])
@pytest.mark.parametrize("entrypoint", ["forward", "text"])
def test_preserves_runner_execution_mode(monkeypatch: pytest.MonkeyPatch, inference_mode: bool, entrypoint: str):
    """HSDP needs version counters; native execution keeps its inference mode."""
    pipeline = object.__new__(SenseNovaU1Pipeline)
    torch.nn.Module.__init__(pipeline)

    class ContextObservedError(Exception):
        pass

    def observe_context() -> None:
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled() is inference_mode
        raise ContextObservedError

    class Parameters:
        @property
        def extra_args(self):
            observe_context()

    monkeypatch.setattr(pipeline, "_is_warmup_request", lambda request: False)
    monkeypatch.setattr(pipeline, "_parse_request", lambda request: observe_context())
    with torch.inference_mode(inference_mode), pytest.raises(ContextObservedError):
        if entrypoint == "forward":
            pipeline.forward(None)
        else:
            pipeline._forward_text(Parameters(), None)
