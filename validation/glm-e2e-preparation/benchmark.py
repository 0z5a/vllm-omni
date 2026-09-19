"""GLM-Image full AR→DiT→VAE comparison with fixed stage placement."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-vae", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.probe:
        os.environ["GLM_PROBE_DIR"] = str(args.output)
    started = time.perf_counter()
    engine = Omni(model=args.model, deploy_config=args.deploy_config, init_timeout=1200, stage_init_timeout=1200)
    ready_seconds = time.perf_counter() - started
    cases = ((512, 512, False), (768, 512, True), (1152, 1024, False), (512, 512, False))
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height, edit) in enumerate(cases[:1] if args.smoke else cases):
            prompt = {
                "prompt": "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
                "modalities": ["image"],
                "mm_processor_kwargs": {"target_h": height, "target_w": width},
            }
            if edit:
                image = Image.new("RGB", (width, height), "lightblue")
                ImageDraw.Draw(image).rectangle((width // 4, height // 4, width // 2, height // 2), fill="red")
                prompt["multi_modal_data"] = {"image": image}
                prompt["prompt"] = "Change the red square to green while preserving the background."
                image.save(args.output / f"input-{case}.png")
            for iteration in range(1 if args.smoke or args.probe else 7):
                ar_params = clone_sampling_params(engine.default_sampling_params_list[0])
                assert isinstance(ar_params, SamplingParams)
                ar_params.seed = 142
                ar_params.extra_args = {"target_h": height, "target_w": width}
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=50,
                    guidance_scale=1.5,
                    seed=142,
                    num_outputs_per_prompt=1,
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[ar_params, params])
                seconds = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1 and images[0].size == (width, height)
                path = args.output / f"{case}-{iteration}.png"
                images[0].save(path)
                row = {
                    "case": case,
                    "width": width,
                    "height": height,
                    "edit": edit,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "parallel_vae": args.parallel_vae,
                    "probe": args.probe,
                    "source": "635df16" if args.parallel_vae else "698f716",
                    "checkpoint_revision": "2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4",
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                records.write(json.dumps(row) + "\n")
                records.flush()
                print(json.dumps(row), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
