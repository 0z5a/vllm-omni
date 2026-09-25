# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ming's per-rank conditions must match the branch selected by CFG parallel."""

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_image import pipeline as ming_pipeline
from vllm_omni.diffusion.models.z_image import pipeline_z_image
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class _Condition(torch.nn.Module):
    def forward(self, query: torch.Tensor, direct: torch.Tensor):
        return query, direct


def test_layer_cfg_ranks_receive_separate_direct_conditions(monkeypatch):
    pipeline = ming_pipeline.MingImageDiffusionPipeline.__new__(ming_pipeline.MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(dtype=torch.bfloat16)
    pipeline.conditioning = _Condition()
    pipeline.is_layer_decomposition = True
    pipeline.cfg_parallel_size = 2
    pipeline.default_num_inference_steps = 12
    pipeline.default_guidance_scale = 2.0
    pipeline._num_frames_per_prompt = 1
    pipeline._pending_prompt_embeds = None
    pipeline._pending_negative_prompt_embeds = None
    pipeline._encode_reference = lambda reference, height, width: reference
    pipeline._decode_latent_frames = lambda latent: latent

    reference = torch.ones(1, 4, 1, 2, 2)
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt={
                    "extra": {
                        "query_hidden_states": torch.ones(256, 4),
                        "direct_hidden_states": torch.ones(1, 6),
                        "reference_image": reference,
                    }
                },
                sampling_params=OmniDiffusionSamplingParams(extra_args={"num_layers": 1, "cfg": 2.0, "seed": 42}),
                request_id="cfg-parallel",
            )
        ]
    )
    seen = []

    def capture_forward(self, req):
        context = get_forward_context()
        seen.append((context.direct_condition.clone(), context.ref_latent.clone()))
        return DiffusionOutput(output=torch.zeros(1, 4, 1, 2, 2))

    monkeypatch.setattr(ZImagePipeline, "forward", capture_forward)
    with set_forward_context(omni_diffusion_config=OmniDiffusionConfig(model=None)):
        for rank in (0, 1):
            monkeypatch.setattr(ming_pipeline, "get_classifier_free_guidance_rank", lambda: rank)
            pipeline.forward(request)

    assert len(seen) == 2
    assert seen[0][0].shape == seen[1][0].shape == (1, 1, 6)
    assert torch.equal(seen[0][0], torch.ones_like(seen[0][0]))
    assert torch.equal(seen[1][0], torch.zeros_like(seen[1][0]))
    assert all(torch.equal(rank_ref, reference) for _, rank_ref in seen)


def test_cfg_parallel_denoising_matches_batched_cfg(monkeypatch):
    class Transformer(torch.nn.Module):
        in_channels = 4

        def forward(self, latents, timestep, conditions):
            return [torch.full_like(latent, condition.item()) for latent, condition in zip(latents, conditions)], None

    class Group:
        def broadcast(self, tensor, src):
            assert src == 0

        def all_gather(self, local, separate_tensors):
            assert separate_tensors
            assert local.shape[0] == 1
            assert local[0, 0, 0, 0, 0].item() == (3 if rank == 0 else 1)
            return [torch.full_like(local, 3), torch.full_like(local, 1)]

    def pipeline(cfg_parallel_size):
        instance = ZImagePipeline.__new__(ZImagePipeline)
        torch.nn.Module.__init__(instance)
        instance.od_config = SimpleNamespace(dtype=torch.float32)
        instance._execution_device = torch.device("cpu")
        instance.vae_scale_factor = 8
        instance.cfg_parallel_size = cfg_parallel_size
        instance.transformer = Transformer()
        instance.scheduler = pipeline_z_image.FlowMatchEulerDiscreteScheduler()
        instance.encode_prompt = lambda **kwargs: ([torch.tensor(3.0)], [torch.tensor(1.0)])
        instance.prepare_latents = lambda *args: torch.zeros(1, 4, 1, 2, 2)
        return instance

    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="",
                sampling_params=OmniDiffusionSamplingParams(
                    height=16,
                    width=16,
                    num_inference_steps=1,
                    guidance_scale=2.0,
                    output_type="latent",
                ),
                request_id="cfg-denoise",
            )
        ]
    )
    baseline = pipeline(1).forward(request).output
    group = Group()
    monkeypatch.setattr(pipeline_z_image, "get_cfg_group", lambda: group)
    for rank in (0, 1):
        monkeypatch.setattr(pipeline_z_image, "get_classifier_free_guidance_rank", lambda: rank)
        parallel = pipeline(2).forward(request).output
        torch.testing.assert_close(parallel, baseline)
