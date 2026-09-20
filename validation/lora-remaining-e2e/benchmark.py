# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Full trained-LoRA E2E for offload, VAE parallelism, and online FP8."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

FEATURES = ("native", "layer", "module", "fp8", "u2-native", "u2-vae2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", choices=FEATURES, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["LORA_REMAINING_PROBE_DIR"] = str(args.output.resolve())
    os.environ["LORA_REMAINING_FEATURE"] = args.feature

    two_rank = args.feature.startswith("u2-")
    parallel = DiffusionParallelConfig(
        ulysses_degree=2 if two_rank else 1,
        vae_patch_parallel_size=2 if args.feature == "u2-vae2" else 1,
    )
    offload = None
    if args.feature == "layer":
        offload = {"mode": "layer", "components": ["dit"], "pin_memory": True}
    elif args.feature == "module":
        offload = {"mode": "module", "components": ["dit", "text_encoder"], "pin_memory": True}

    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=parallel,
        diffusion_offload_config=offload,
        quantization_config="fp8" if args.feature == "fp8" else None,
        vae_use_tiling=two_rank,
        diffusion_model_runner_cls="probe.ProbeRunner" if args.probe else None,
        lora_backend="peft",
        max_cpu_loras=2,
        dtype="bfloat16",
        enforce_eager=True,
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    adapters = {
        "A": LoRARequest("helio", 1, args.adapter_a),
        "B": LoRARequest("fuli", 2, args.adapter_b),
        "None": None,
    }
    sequence = (
        ("A", 1.0),
        ("A", 1.0),
        ("B", 1.0),
        ("None", 1.0),
        ("A", 1.0),
        ("A", 0.0),
        ("B", 0.0),
        ("A", 1.0),
    )
    cases = (
        (512, 512, "A red ceramic teapot beside two blue cups on a wooden table, soft daylight."),
        (768, 512, "A green bicycle beside a brick wall, afternoon sunshine."),
        (512, 512, "A red ceramic teapot beside two blue cups on a wooden table, soft daylight."),
    )
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height, prompt) in enumerate(cases):
            for cycle in range(1 if args.probe else 2 + args.repeats):
                for position, (adapter, scale) in enumerate(sequence):
                    params = OmniDiffusionSamplingParams(
                        width=width,
                        height=height,
                        num_inference_steps=9,
                        guidance_scale=0.0,
                        seed=142,
                        num_outputs_per_prompt=1,
                        lora_request=adapters[adapter],
                        lora_scale=scale,
                    )
                    started = time.perf_counter()
                    outputs = engine.generate(
                        {"prompt": prompt, "modalities": ["image"]},
                        sampling_params_list=[params],
                    )
                    seconds = time.perf_counter() - started
                    images = extract_images_from_outputs(outputs)
                    assert len(images) == 1 and images[0].size == (width, height)
                    path = args.output / f"case-{case}-cycle-{cycle}-position-{position}.png"
                    images[0].save(path)
                    record = {
                        "case": case,
                        "cycle": cycle,
                        "position": position,
                        "adapter": adapter,
                        "scale": scale,
                        "width": width,
                        "height": height,
                        "feature": args.feature,
                        "ranks": 2 if two_rank else 1,
                        "warmup": cycle < 2,
                        "probe": args.probe,
                        "seconds": seconds,
                        "ready_seconds": ready_seconds,
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    records.write(json.dumps(record) + "\n")
                    records.flush()
                    print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
