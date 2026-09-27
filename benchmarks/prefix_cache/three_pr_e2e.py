# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-model prefix-cache E2E for the three cache PR heads."""

import argparse
import asyncio
import json
import os
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

import psutil
import yaml
from transformers import AutoTokenizer
from vllm.model_executor.models.gemma import GemmaForCausalLM
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView
from vllm_omni.entrypoints.omni import Omni

original_gemma_forward = GemmaForCausalLM.forward


def gemma_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **_omni_metadata):
    return original_gemma_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)


GemmaForCausalLM.forward = gemma_forward
original_qwen_forward = Qwen2ForCausalLM.forward


def qwen_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **_omni_metadata):
    return original_qwen_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)


Qwen2ForCausalLM.forward = qwen_forward
original_opt_forward = OPTForCausalLM.forward


def opt_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **_omni_metadata):
    return original_opt_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)


OPTForCausalLM.forward = opt_forward


def register_text_pipeline(model_type: str, model_arch: str) -> None:
    register_pipeline(
        PipelineConfig(
            model_type=model_type,
            model_arch=model_arch,
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage=model_type,
                    execution_type=StageExecutionType.LLM_AR,
                    input_sources=(),
                    final_output=True,
                    final_output_type="text",
                    engine_output_type="text",
                    owns_tokenizer=True,
                ),
            ),
        )
    )


register_text_pipeline("gemma3_text", "Gemma3ForCausalLM")


original_view_init = FullAttentionGroupView.__init__
verified_layouts = set()


def record_layout_8051(view, batch, block_size):
    original_view_init(view, batch, block_size)
    table = batch.block_table[0]
    layout = (0, table.kv_cache_block_size, table.block_size, table.blocks_per_kv_block, 1)
    if layout not in verified_layouts:
        verified_layouts.add(layout)
        print("VERIFIED_BLOCK_LAYOUT " + json.dumps(layout), flush=True)


def record_layout_8056(view, batch, block_size, group_id=0):
    original_view_init(view, batch, block_size, group_id)
    table = batch.block_table[group_id]
    layout = (group_id, table.kv_cache_block_size, table.block_size, table.blocks_per_kv_block, 1)
    if layout not in verified_layouts:
        verified_layouts.add(layout)
        print("VERIFIED_BLOCK_LAYOUT " + json.dumps(layout), flush=True)


def record_layout_8066(view, batch, block_size, group_id=0, dcp_world_size=1):
    original_view_init(view, batch, block_size, group_id, dcp_world_size)
    table = batch.block_table[group_id]
    layout = (group_id, table.kv_cache_block_size, table.block_size, table.blocks_per_kv_block, table.dcp_world_size)
    if layout not in verified_layouts:
        verified_layouts.add(layout)
        print("VERIFIED_BLOCK_LAYOUT " + json.dumps(layout), flush=True)


milestone = os.environ.get("OMNI_CACHE_E2E_MILESTONE")
if milestone is not None:
    FullAttentionGroupView.__init__ = {
        "8051": record_layout_8051,
        "8056": record_layout_8056,
        "8066": record_layout_8066,
    }[milestone]


def request_benchmark_exit(core: EngineCoreProc) -> None:
    assert not core.has_work()
    core.shutdown_state = EngineShutdownState.REQUESTED


setattr(EngineCoreProc, "request_benchmark_exit", request_benchmark_exit)
original_kill = os.kill


def signal_guard(pid: int, sig: int) -> None:
    if sig:
        raise RuntimeError(f"Process signals are prohibited: pid={pid}, signal={sig}")
    original_kill(pid, sig)


os.kill = signal_guard


def prohibit_process_signal(self) -> None:
    raise RuntimeError(f"Process termination is prohibited: pid={self.pid}")


BaseProcess.terminate = prohibit_process_signal
BaseProcess.kill = prohibit_process_signal
psutil.Process.terminate = prohibit_process_signal
psutil.Process.kill = prohibit_process_signal


def natural_shutdown(omni: Omni) -> None:
    clients = [client for pool in omni.engine.stage_pools for client in pool.clients if client is not None]
    for client in clients:
        client.resources.engine_manager.manager_stopped.set()
        client.resources.engine_manager._finalizer.detach()
    for client in clients:
        loop = client.resources.output_queue_task.get_loop()
        asyncio.run_coroutine_threadsafe(client.call_utility_async("request_benchmark_exit"), loop)
    for client in clients:
        for process in client.resources.engine_manager.processes:
            process.join(60)
            assert process.exitcode == 0, (process.pid, process.exitcode)
    omni.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=int, choices=[0, 1], required=True)
    parser.add_argument("--milestone", type=int, choices=[8051, 8056, 8066], required=True)
    parser.add_argument("--requests", type=int, default=7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dcp-world-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attention-backend")
    parser.add_argument("--memory", type=float, default=0.4)
    parser.add_argument("--prompt-tokens", type=int, default=960)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--log-stats", action="store_true")
    args = parser.parse_args()
    assert str(args.milestone) == milestone
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    model_type = cfg["model_type"]
    if model_type == "gemma3_text":
        layer_types = cfg.get("layer_types")
        if layer_types is None:
            assert cfg["sliding_window_pattern"] > 1 and cfg["sliding_window"] > 0
        else:
            assert "full_attention" in layer_types and "sliding_attention" in layer_types
    elif model_type == "gemma":
        assert cfg["num_key_value_heads"] == 1
        register_text_pipeline(model_type, cfg["architectures"][0])
    elif model_type == "opt":
        assert cfg["num_attention_heads"] % args.dcp_world_size == 0
        register_text_pipeline(model_type, cfg["architectures"][0])
    else:
        assert model_type == "qwen2" and not cfg["use_sliding_window"]
        register_text_pipeline(model_type, cfg["architectures"][0])
    assert args.dcp_world_size == 1 or len(args.device.split(",")) == args.dcp_world_size
    extras = dict(block_size=args.block_size)
    if args.attention_backend:
        extras["attention_config"] = dict(backend=args.attention_backend)
    if args.dcp_world_size > 1:
        assert args.attention_backend == "FLASHINFER"
        extras["decode_context_parallel_size"] = args.dcp_world_size
    deploy = args.output / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            dict(
                stages=[
                    dict(
                        stage_id=0,
                        devices=args.device,
                        max_num_seqs=1,
                        max_model_len=max(2048, args.prompt_tokens + args.max_tokens),
                        max_num_batched_tokens=max(1024, args.prompt_tokens),
                        gpu_memory_utilization=args.memory,
                        enforce_eager=True,
                        async_scheduling=False,
                        enable_prefix_caching=bool(args.cache),
                        tensor_parallel_size=args.dcp_world_size,
                        engine_extras=extras,
                    )
                ]
            )
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    context = "The garden contains flowers, birds, and a quiet path beside a small pond. " * (
        args.prompt_tokens // 8 + 1
    )
    prompt_tokens = tokenizer.encode(context)[: args.prompt_tokens]
    assert len(prompt_tokens) == args.prompt_tokens
    if model_type == "gemma3_text":
        assert len(prompt_tokens) > cfg["sliding_window"]
    omni = Omni(
        model=args.model, deploy_config=str(deploy), log_stats=args.log_stats, stage_init_timeout=900, init_timeout=1800
    )
    sampling = [SamplingParams(temperature=0, seed=42, max_tokens=args.max_tokens)]
    for index in range(args.requests + 1):
        start = time.perf_counter()
        outputs = list(
            omni.generate([dict(prompt_token_ids=prompt_tokens)], sampling, py_generator=True, use_tqdm=False)
        )
        assert len(outputs) == 1 and outputs[0].finished
        result = outputs[0]
        row = dict(
            index=index,
            warmup=index == 0,
            seconds=time.perf_counter() - start,
            prompt_tokens=len(prompt_tokens),
            cached_tokens=result.num_cached_tokens,
            token_ids=list(result.outputs[0].token_ids),
            text=result.outputs[0].text,
        )
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    natural_shutdown(omni)
    (args.output / "complete").write_text("all stages exited naturally\n")


if __name__ == "__main__":
    main()
