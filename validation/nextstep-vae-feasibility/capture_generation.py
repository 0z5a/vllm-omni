"""Run one complete NextStep request and preserve its real AR latent."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        diffusion_model_runner_cls="capture_runner.LatentCaptureRunner",
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=7200,
        stage_init_timeout=7200,
    )
    prompt = {
        "prompt": "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
        "negative_prompt": "blurry, distorted",
        "modalities": ["image"],
    }
    params = OmniDiffusionSamplingParams(
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=4.0,
        seed=142,
        num_outputs_per_prompt=1,
    )
    started = time.perf_counter()
    outputs = engine.generate(prompt, sampling_params_list=[params])
    seconds = time.perf_counter() - started
    images = extract_images_from_outputs(outputs)
    assert len(images) == 1 and images[0].size == (512, 512)
    image_path = args.output / "full-generation.png"
    images[0].save(image_path)
    latent_path = args.output / "ar-latent.pt"
    assert latent_path.is_file()
    print(
        json.dumps(
            {
                "seconds": seconds,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "latent_sha256": hashlib.sha256(latent_path.read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )
    engine.close()


if __name__ == "__main__":
    main()
