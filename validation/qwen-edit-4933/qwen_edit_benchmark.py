"""Raw-image Qwen Edit/2509 E2E for the existing PR 4933 runtime patch."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plus", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--parallel-vae", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["QWEN_PROBE_DIR"] = str(args.output.resolve())
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
            vae_patch_parallel_size=2 if args.parallel_vae else 1,
        ),
        diffusion_model_runner_cls="qwen_edit_probe.ConditioningProbeRunner" if args.probe else None,
        vae_use_tiling=True,
        vae_use_slicing=True,
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    with (args.output / "results.jsonl").open("w") as records:
        cases = ((512, 512, 1), (768, 512, 1), (512, 512, 1))
        if args.plus:
            cases = ((512, 512, 1), (768, 512, 2), (512, 768, 3), (512, 512, 1))
        for case, (width, height, count) in enumerate(cases):
            references = []
            for index in range(count):
                if index == 2:
                    image = references[0].copy()
                else:
                    image_width, image_height = (width, height) if index == 0 else (height, width)
                    image = Image.new("RGB", (image_width, image_height), "white")
                    ImageDraw.Draw(image).rectangle(
                        (image_width // 4, image_height // 4, image_width * 3 // 4, image_height * 3 // 4),
                        fill=("red", "blue")[index],
                    )
                image.save(args.output / f"input-{case}-{index}.png")
                references.append(image)
            prompt = {
                "prompt": "Change the red square to a green square, preserving the white background.",
                "negative_prompt": "",
                "modalities": ["image"],
                "multi_modal_data": {"image": references if args.plus else references[0]},
            }
            for iteration in range(1 if args.probe else 7):
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=50,
                    true_cfg_scale=4.0,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                seconds = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1 and images[0].size == (width, height)
                path = args.output / f"case-{case}-{iteration}.png"
                images[0].save(path)
                row = {
                    "base_sha": "698f7160125d3071b8c0eef69b1b03fa8dfba766",
                    "runtime_patch": "PR4933@37e4407dcf10d49ab9215dfd4a545fb9f014b5c4" if args.parallel_vae else None,
                    "probe": args.probe,
                    "parallel_vae": args.parallel_vae,
                    "plus": args.plus,
                    "case": case,
                    "width": width,
                    "height": height,
                    "references": count,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "input_sizes": [image.size for image in references],
                    "input_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in references],
                }
                records.write(json.dumps(row) + "\n")
                records.flush()
                print(json.dumps(row), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
