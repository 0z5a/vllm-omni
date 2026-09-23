# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a repeatable three-stage text/audio prefix-cache workload.

Run once with ``--cache 0`` and once with ``--cache 1``, using different
output directories. Compare them with ``compare_prefix_cache_split_e2e.py``.
Shutdown uses a utility request and never sends process signals.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

import psutil
import torch
import yaml
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView
from vllm_omni.entrypoints.omni import Omni


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
original_view_init = FullAttentionGroupView.__init__
verified_layouts = set()


def record_layout(view, batch, block_size):
    original_view_init(view, batch, block_size)
    table = batch.block_table[0]
    layout = (table.kv_cache_block_size, table.block_size, table.blocks_per_kv_block)
    if layout not in verified_layouts:
        verified_layouts.add(layout)
        record = dict(pid=os.getpid(), allocator=layout[0], kernel=layout[1], split=layout[2])
        print("VERIFIED_BLOCK_LAYOUT " + json.dumps(record), flush=True)


FullAttentionGroupView.__init__ = record_layout


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=int, choices=[0, 1], required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--memory", default="0.55,0.55,0.15")
    parser.add_argument("--context-repeat", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = args.devices.split(",")
    assert len(devices) == 3
    stage_memory = tuple(float(value) for value in args.memory.split(","))
    assert len(stage_memory) == 3
    stages = []
    for stage_id in range(3):
        stage = dict(
            stage_id=stage_id,
            devices=devices[stage_id],
            max_num_seqs=1,
            max_model_len=4096,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=stage_memory[stage_id],
            enforce_eager=True,
            trust_remote_code=True,
            async_scheduling=False,
            enable_flashinfer_autotune=False,
            enable_prefix_caching=bool(args.cache and stage_id == 0),
        )
        if stage_id == 0:
            stage["engine_extras"] = dict(block_size=args.block_size, attention_config=dict(backend="FLASHINFER"))
        stages.append(stage)
    deploy = args.output / "deploy.yaml"
    deploy.write_text(yaml.safe_dump(dict(async_chunk=False, stages=stages)))
    omni = Omni(model=args.model, deploy_config=str(deploy), log_stats=True, stage_init_timeout=900, init_timeout=1800)
    params = [
        SamplingParams(temperature=0, seed=42, max_tokens=64),
        SamplingParams(temperature=0, seed=42, max_tokens=512, stop_token_ids=[8294]),
        SamplingParams(temperature=0, seed=42, max_tokens=2048),
    ]
    context = "The following context describes a quiet garden with trees, flowers, and birds. " * args.context_repeat
    prompt = (
        "<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        "<|im_end|>\n<|im_start|>user\n"
        + context
        + "\nSay exactly: Welcome to the quiet garden.<|im_end|>\n<|im_start|>assistant\n"
    )
    for index in range(args.requests + 1):
        start = time.perf_counter()
        outputs = []
        stage_seconds = {}
        for output in omni.generate(
            [dict(prompt=prompt, modalities=["text", "audio"])], params, py_generator=True, use_tqdm=False
        ):
            outputs.append(output)
            stage_seconds[output.final_output_type] = time.perf_counter() - start
        elapsed = time.perf_counter() - start
        row = dict(
            index=index,
            warmup=index == 0,
            seconds=elapsed,
            stage_seconds=stage_seconds,
            cache=bool(args.cache),
            stages=[],
        )
        for output in outputs:
            item = output.outputs[0]
            stage = dict(
                stage_id=output.stage_id,
                kind=output.final_output_type,
                token_ids=list(item.token_ids),
                cached_tokens=output.num_cached_tokens,
                finish_reason=item.finish_reason,
            )
            if output.stage_id == 0:
                stage["prompt_tokens"] = len(output.prompt_token_ids or [])
            if output.final_output_type == "audio":
                audio = item.multimodal_output["audio"].detach().cpu()
                assert torch.isfinite(audio).all() and audio.numel() > 0
                torch.save(audio, args.output / f"audio-{index}.pt")
                stage.update(
                    audio_samples=audio.numel(),
                    audio_seconds=audio.numel() / 24000,
                    audio_sha256=hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
                )
            row["stages"].append(stage)
        assert any(stage["kind"] == "audio" for stage in row["stages"])
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    natural_shutdown(omni)
    (args.output / "complete").write_text("all stages exited naturally\n")


if __name__ == "__main__":
    main()
