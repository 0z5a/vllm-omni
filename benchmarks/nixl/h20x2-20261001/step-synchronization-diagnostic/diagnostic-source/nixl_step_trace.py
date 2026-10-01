"""Finite per-step checkpoint diagnosis, kept outside the production path."""

import hashlib
import json
import os
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import torch
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionAdapter,
)
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import (
    DistributedAutoencoderKLHunyuan,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Text2ImagePipeline,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3FlowMatchEulerDiscreteScheduler,
    HunyuanImage3Pipeline,
)

ROOT = Path(os.environ["NIXL_TRACE_DIRECTORY"])
ROOT.mkdir(parents=True, exist_ok=True)
STATE = {"request": -1, "scheduler": 0}


def snapshot(tensor):
    return tensor.detach().cpu().clone()


def digest(tensor):
    data = snapshot(tensor).contiguous()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sha256": hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(data).all()),
        "min": float(data.min()),
        "max": float(data.max()),
        "rms": float(data.float().square().mean().sqrt()),
    }


def save(name, tensors):
    torch.save(tensors, ROOT / f"request-{STATE['request']}-{name}.pt")


def install():
    call = HunyuanImage3Text2ImagePipeline.__call__
    forward = HunyuanImage3Pipeline.forward_call
    step = HunyuanImage3FlowMatchEulerDiscreteScheduler.step
    decode = DistributedAutoencoderKLHunyuan.decode
    context = DiffusionPagedAttentionAdapter.prepare_layer_context

    @wraps(call)
    def traced_call(self, *args, **kwargs):
        STATE["request"] += 1
        STATE["scheduler"] = 0
        output = call(self, *args, **kwargs)
        save("pipeline-output", {"image": snapshot(output[0])})
        return output

    @wraps(forward)
    def traced_forward(self, *args, **kwargs):
        output = forward(self, *args, **kwargs)
        prediction = output.diffusion_prediction
        if prediction is not None:
            index = get_forward_context().denoise_step_idx
            save(f"prediction-{index:02d}", {"prediction": snapshot(prediction)})
        return output

    @wraps(step)
    def traced_step(self, model_output, timestep, sample, *args, **kwargs):
        index = STATE["scheduler"]
        before = snapshot(sample)
        result = step(self, model_output, timestep, sample, *args, **kwargs)
        save(
            f"scheduler-{index:02d}",
            {
                "input": before,
                "prediction": snapshot(model_output),
                "output": snapshot(result[0]),
                "timestep": snapshot(timestep),
                "step_index_after": self.step_index,
                "sigmas": snapshot(
                    self.sigmas[self.step_index - 1 : self.step_index + 1]
                ),
            },
        )
        STATE["scheduler"] += 1
        return result

    @wraps(decode)
    def traced_decode(self, z, *args, **kwargs):
        before = snapshot(z)
        result = decode(self, z, *args, **kwargs)
        save("vae", {"input": before, "output": snapshot(result[0])})
        return result

    @wraps(context)
    def traced_context(self, layer_name, query, key, value, **kwargs):
        ctx = context(self, layer_name, query, key, value, **kwargs)
        layer = int(layer_name.split(".layers.")[1].split(".")[0])
        if layer in (0, 31):
            batch = self._active_batch
            index = get_forward_context().denoise_step_idx
            record = {
                "request": STATE["request"],
                "step": index,
                "layer": layer,
                "fa_version": ctx.layer.impl.vllm_flash_attn_version,
                "rows": [asdict(row) for row in batch.rows],
                "query": digest(query),
                "key": digest(key),
                "value": digest(value),
            }
            with (ROOT / "attention.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            caches = [
                snapshot(ctx.layer.kv_cache.index_select(0, blocks[:2].long()))
                for blocks in batch.block_tables[0]
            ]
            phase = "negative" if index is None else f"{index:02d}"
            metadata = [
                {
                    "query_start_loc": snapshot(item.query_start_loc),
                    "seq_lens": snapshot(item.seq_lens),
                    "block_table": snapshot(item.block_table),
                    "scheduler_metadata": snapshot(item.scheduler_metadata)
                    if item.scheduler_metadata is not None
                    else None,
                    "causal": item.causal,
                    "max_num_splits": item.max_num_splits,
                }
                for item in ctx.piecewise_native_metadata
            ]
            save(
                f"prefix-{phase}-layer-{layer}",
                {
                    "rows": record["rows"],
                    "caches": caches,
                    "native_metadata": metadata,
                    "slot_mapping": snapshot(ctx.slot_mapping),
                },
            )
        return ctx

    HunyuanImage3Text2ImagePipeline.__call__ = traced_call
    HunyuanImage3Pipeline.forward_call = traced_forward
    HunyuanImage3FlowMatchEulerDiscreteScheduler.step = traced_step
    DistributedAutoencoderKLHunyuan.decode = traced_decode
    DiffusionPagedAttentionAdapter.prepare_layer_context = traced_context
