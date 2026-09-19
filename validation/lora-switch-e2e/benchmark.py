"""Real trained-adapter switching, one fresh process per feature arm."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature", choices=("none", "tea_cache", "cache_dit", "ulysses", "ring", "tp", "hsdp"), required=True
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["LORA_PROBE_DIR"] = str(args.output.resolve())
    parallel = DiffusionParallelConfig(
        tensor_parallel_size=2 if args.feature == "tp" else 1,
        ulysses_degree=2 if args.feature == "ulysses" else 1,
        ring_degree=2 if args.feature == "ring" else 1,
        use_hsdp=args.feature == "hsdp",
        hsdp_shard_size=2 if args.feature == "hsdp" else -1,
    )
    cache = args.feature if args.feature in ("tea_cache", "cache_dit") else None
    cache_config = (
        {"rel_l1_thresh": 0.2}
        if cache == "tea_cache"
        else {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 2,
            "max_continuous_cached_steps": 3,
            "residual_diff_threshold": 0.24,
            "enable_taylorseer": False,
            "scm_steps_mask_policy": None,
        }
    )
    engine = Omni(
        model=args.model,
        parallel_config=parallel,
        cache_backend=cache,
        cache_config=cache_config if cache else None,
        diffusion_model_runner_cls="probe.ProbeRunner" if args.probe else None,
        lora_backend="peft",
        max_cpu_loras=2,
        dtype="bfloat16",
        enforce_eager=True,
        init_timeout=900,
        stage_init_timeout=900,
    )
    adapters = {"A": LoRARequest("helio", 1, args.adapter_a), "B": LoRARequest("fuli", 2, args.adapter_b), "None": None}
    sequence = (("A", 1.0), ("A", 1.0), ("B", 1.0), ("None", 1.0), ("A", 1.0), ("A", 0.0), ("B", 0.0), ("A", 1.0))
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height) in enumerate(((512, 512), (768, 512), (512, 512))):
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
                        {
                            "prompt": "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
                            "modalities": ["image"],
                        },
                        sampling_params_list=[params],
                    )
                    seconds = time.perf_counter() - started
                    images = extract_images_from_outputs(outputs)
                    assert len(images) == 1 and images[0].size == (width, height)
                    path = args.output / f"case-{case}-cycle-{cycle}-position-{position}.png"
                    images[0].save(path)
                    record = dict(
                        case=case,
                        cycle=cycle,
                        position=position,
                        adapter=adapter,
                        scale=scale,
                        width=width,
                        height=height,
                        feature=args.feature,
                        warmup=cycle < 2,
                        probe=args.probe,
                        seconds=seconds,
                        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    records.write(json.dumps(record) + "\n")
                    records.flush()
                    print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
