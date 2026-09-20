"""Full Qwen Edit LoRA switch sequence, with fresh CFG1/CFG2 processes."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

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
    parser.add_argument("--cfg-degree", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["LORA_PROBE_DIR"] = str(args.output.resolve())
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=args.cfg_degree),
        diffusion_model_runner_cls="probe.CFGProbeRunner" if args.probe else None,
        enable_cpu_offload=True,
        vae_use_tiling=True,
        vae_use_slicing=True,
        lora_backend="peft",
        max_cpu_loras=2,
        dtype="bfloat16",
        enforce_eager=True,
        init_timeout=1800,
        stage_init_timeout=1800,
    )
    ready_seconds = time.perf_counter() - started
    adapters = {
        "A": LoRARequest("edit-r1", 1, args.adapter_a),
        "B": LoRARequest("cocoedit", 2, args.adapter_b),
        "None": None,
    }
    sequence = (("A", 1.0), ("A", 1.0), ("B", 1.0), ("None", 1.0), ("A", 1.0), ("A", 0.0), ("B", 0.0), ("A", 1.0))
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height) in enumerate(((512, 512), (768, 512), (512, 512))):
            reference = Image.new("RGB", (width, height), "white")
            ImageDraw.Draw(reference).rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill="red")
            reference.save(args.output / f"input-{case}.png")
            input_sha256 = hashlib.sha256(reference.tobytes()).hexdigest()
            prompt = {
                "prompt": "Change the red square to a green square, preserving the white background.",
                "negative_prompt": "blurry, distorted shapes",
                "modalities": ["image"],
                "multi_modal_data": {"image": [reference]},
            }
            for cycle in range(1 if args.probe else 7):
                for position, (adapter, scale) in enumerate(sequence):
                    params = OmniDiffusionSamplingParams(
                        width=width,
                        height=height,
                        num_inference_steps=50,
                        true_cfg_scale=4.0,
                        seed=142,
                        num_outputs_per_prompt=1,
                        lora_request=adapters[adapter],
                        lora_scale=scale,
                    )
                    started = time.perf_counter()
                    outputs = engine.generate(prompt, sampling_params_list=[params])
                    seconds = time.perf_counter() - started
                    images = extract_images_from_outputs(outputs)
                    assert len(images) == 1 and images[0].size == (width, height)
                    path = args.output / f"case-{case}-cycle-{cycle}-position-{position}.png"
                    images[0].save(path)
                    record = dict(
                        source_sha="3ea51522ba780697f3670cadfdb117781b3ea7f3",
                        case=case,
                        cycle=cycle,
                        position=position,
                        adapter=adapter,
                        scale=scale,
                        width=width,
                        height=height,
                        feature="cfg",
                        cfg_degree=args.cfg_degree,
                        warmup=cycle < 2,
                        probe=args.probe,
                        seconds=seconds,
                        ready_seconds=ready_seconds,
                        input_sha256=input_sha256,
                        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    records.write(json.dumps(record) + "\n")
                    records.flush()
                    print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
