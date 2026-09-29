# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure repeated YuE2 text-to-audio requests in one engine process."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from examples.offline_inference.yue2.end2end import sampling_params
from vllm_omni import Omni
from vllm_omni.model_executor.models.yue2.constants import KEY_TEMPERATURE
from vllm_omni.model_executor.models.yue2.prompt import semantic_prefix_ids
from vllm_omni.model_executor.models.yue2.tokenizer import YuE2TextTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = YuE2TextTokenizer(Path(args.model) / "qwen.tiktoken")
    prompt_ids = semantic_prefix_ids(tokenizer.encode, "gentle piano, 85 BPM", "[Verse] Hello, stay a while", "off")
    engine = Omni(
        model=args.model,
        trust_remote_code=True,
        deploy_config=args.deploy_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    for index in range(args.requests + 1):
        params = sampling_params(
            engine, phase="semantic", seed=831001, max_frames=args.max_frames, prompt_ids=prompt_ids
        )
        if args.greedy:
            params.extra_args[KEY_TEMPERATURE] = 0.0
        start = time.perf_counter()
        output = engine.generate([{"prompt_token_ids": prompt_ids}], [params])[0].outputs[0]
        wall_s = time.perf_counter() - start
        audio = output.multimodal_output["audio"].float().cpu()
        if not torch.isfinite(audio).all():
            raise ValueError("nonfinite audio")
        tokens = list(output.token_ids)
        print(
            json.dumps(
                {
                    "request": index,
                    "warmup": index == 0,
                    "wall_s": wall_s,
                    "tokens": len(tokens),
                    "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    "generated_token_ids": tokens,
                    "audio_s": audio.shape[-1] / 48000,
                    "audio_sha256": hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
