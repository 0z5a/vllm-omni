"""SenseNova full-checkpoint native versus HSDP2; TP remains one."""

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
    parser.add_argument("--hsdp", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(use_hsdp=args.hsdp, hsdp_shard_size=2 if args.hsdp else 1),
        enforce_eager=True,
        dtype="bfloat16",
        init_timeout=900,
        stage_init_timeout=900,
    )
    ready_seconds = time.perf_counter() - started
    reference = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(reference).rectangle((128, 128, 384, 384), fill="red")
    reference.save(args.output / "input.png")
    with (args.output / "results.jsonl").open("w") as records:
        for case, (width, height, batch, edit) in enumerate(
            ((512, 512, 1, False), (768, 512, 2, False), (512, 512, 1, True), (512, 512, 1, False))
        ):
            for iteration in range(7):
                prompt = {
                    "prompt": "Change the red square to green."
                    if edit
                    else "A red ceramic teapot beside two blue cups on a wooden table, soft daylight.",
                    "modalities": ["image"],
                }
                if edit:
                    prompt["multi_modal_data"] = {"image": reference}
                params = OmniDiffusionSamplingParams(
                    width=width,
                    height=height,
                    num_inference_steps=50,
                    seed=142,
                    num_outputs_per_prompt=1,
                    extra_args={"batch_size": batch, "cfg_scale": 4.0, "img_cfg_scale": 1.0, "think": False},
                )
                started = time.perf_counter()
                outputs = engine.generate(prompt, sampling_params_list=[params])
                seconds = time.perf_counter() - started
                images = extract_images_from_outputs(outputs)
                assert len(images) == batch, (len(images), batch)
                hashes = []
                for index, image in enumerate(images):
                    assert image.size == (width, height)
                    path = args.output / f"case-{case}-{iteration}-{index}.png"
                    image.save(path)
                    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                row = {
                    "source_sha": "e94dafe1d6d39637aebef32babff186e9b4933cb"
                    if args.hsdp
                    else "fb3d6d54be21740398f1c8f824361b3850d9ec67",
                    "hsdp": args.hsdp,
                    "gpus": 2 if args.hsdp else 1,
                    "tp": 1,
                    "case": case,
                    "width": width,
                    "height": height,
                    "batch": batch,
                    "edit": edit,
                    "iteration": iteration,
                    "warmup": iteration < 2,
                    "seconds": seconds,
                    "ready_seconds": ready_seconds,
                    "image_sha256": hashes,
                }
                records.write(json.dumps(row) + "\n")
                records.flush()
                print(json.dumps(row), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
