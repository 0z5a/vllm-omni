# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Share one long prefix across a batch of concurrent requests.

Every in-tree Gemma driver pins ``max_num_seqs=1`` and submits one request at a
time, so a prefix hit can only ever skip prefill for a single request and the
fixed per-request cost hides it. This submits ``--batch`` identical requests in
one ``generate`` call with ``max_num_seqs=--batch``, so a hit skips prefill for
the whole batch. That is the shape where a prefix cache pays off, and it is the
shape the PR's acceptance criteria describe ("two identical requests admitted in
the same step").

Cache off: every round recomputes the whole prompt for every sequence.
Cache on: the first round fills the cache, later rounds reuse it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.entrypoints.omni import Omni

# Gemma 3 is not a registered Omni pipeline; the in-tree driver declares it the
# same way, and the two must agree or the model does not resolve.
register_pipeline(
    PipelineConfig(
        model_type="gemma3_text",
        model_arch="Gemma3ForCausalLM",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="gemma3_text",
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


def request_benchmark_exit(core: EngineCoreProc) -> None:
    assert not core.has_work()
    core.shutdown_state = EngineShutdownState.REQUESTED


setattr(EngineCoreProc, "request_benchmark_exit", request_benchmark_exit)


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
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8, help="identical concurrent requests per round")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--memory", type=float, default=0.4)
    parser.add_argument("--log-stats", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((Path(args.model) / "config.json").read_text())
    assert cfg["model_type"] == "gemma3_text", "this harness targets the hybrid Gemma 3 layout"

    deploy = args.output / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            dict(
                stages=[
                    dict(
                        stage_id=0,
                        devices=args.device,
                        max_num_seqs=args.batch,
                        max_model_len=max(2048, args.prompt_tokens + args.max_tokens),
                        max_num_batched_tokens=max(1024, args.prompt_tokens + args.max_tokens),
                        gpu_memory_utilization=args.memory,
                        enforce_eager=True,
                        async_scheduling=False,
                        enable_prefix_caching=bool(args.cache),
                        engine_extras=dict(block_size=16, attention_config=dict(backend="FLASHINFER")),
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
    assert len(prompt_tokens) > cfg["sliding_window"]

    omni = Omni(
        model=args.model, deploy_config=str(deploy), log_stats=args.log_stats, stage_init_timeout=900, init_timeout=1800
    )
    # The sampling list is per stage, not per request: one stage here, so one
    # entry, and the batch is the request list.
    sampling = [SamplingParams(temperature=0, seed=42, max_tokens=args.max_tokens)]
    requests = [dict(prompt_token_ids=prompt_tokens)] * args.batch

    # Round 0 fills the cache and is never timed.
    for index in range(args.rounds + 1):
        start = time.perf_counter()
        outputs = list(omni.generate(requests, sampling, py_generator=True, use_tqdm=False))
        elapsed = time.perf_counter() - start
        assert len(outputs) == args.batch, (len(outputs), args.batch)
        row = dict(
            index=index,
            warmup=index == 0,
            seconds=elapsed,
            batch=args.batch,
            cache=bool(args.cache),
            prompt_tokens=len(prompt_tokens),
            cached_tokens=[output.num_cached_tokens for output in outputs],
            token_ids=[list(output.outputs[0].token_ids) for output in outputs],
        )
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    natural_shutdown(omni)
    (args.output / "complete").write_text("engine exited naturally\n")


if __name__ == "__main__":
    main()
