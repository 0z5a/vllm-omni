"""Full raw-image Layered E2E, with identical TP2 and native tiled decode."""

import argparse
import hashlib
import json
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
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--parallel-vae", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
            vae_patch_parallel_size=2 if args.parallel_vae else 1,
        ),
        vae_use_tiling=True,
        vae_use_slicing=True,
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    with (args.output / "results.jsonl").open("w") as records:
        for case, (resolution, width, height, layers) in enumerate(
            (
                (640, 640, 640, 2),
                (640, 768, 512, 4),
                (1024, 1024, 1024, 2),
                (640, 640, 640, 2),
            )
        ):
            reference = Image.new("RGBA", (width, height), "white")
            drawing = ImageDraw.Draw(reference)
            drawing.rectangle((width // 8, height // 4, width // 2, height * 3 // 4), fill="red")
            drawing.ellipse((width // 2, height // 3, width * 7 // 8, height * 2 // 3), fill="blue")
            reference.save(args.output / f"input-{case}.png")
            for iteration in range(2 + args.repeats):
                params = OmniDiffusionSamplingParams(
                    resolution=resolution,
                    layers=layers,
                    num_inference_steps=50,
                    true_cfg_scale=4.0,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                prompt = {
                    "prompt": "A red rectangle and a blue ellipse on a white background.",
                    "negative_prompt": "",
                    "modalities": ["image"],
                    "multi_modal_data": {"image": reference},
                }
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                seconds = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == layers, (len(images), layers)
                hashes = []
                for layer, image in enumerate(images):
                    assert image.size == (width, height) and image.mode == "RGBA"
                    path = args.output / f"case-{case}-{iteration}-layer-{layer}.png"
                    image.save(path)
                    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                record = {
                    "source_sha": args.source_sha,
                    "parallel_vae": args.parallel_vae,
                    "case": case,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "layers": layers,
                    "steps": 50,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "input_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                    "layer_sha256": hashes,
                }
                records.write(json.dumps(record) + "\n")
                records.flush()
                print(json.dumps(record), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
