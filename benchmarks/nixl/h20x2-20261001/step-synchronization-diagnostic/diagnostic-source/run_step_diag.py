# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-checkpoint AR -> native paged DiT -> VAE validation on a H20 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from nixl_no_kill import install
from PIL import Image
from vllm import SamplingParams

install()  # Also runs when multiprocessing imports this module in a child.

from nixl_hunyuan_prompt import build_ar_tokens, load_prompt_builder  # noqa: E402

from nixl_step_trace import install as install_trace

install_trace()
from vllm_omni import Omni  # noqa: E402
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (  # noqa: E402
    resolve_stop_token_ids,
)
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs  # noqa: E402
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402

PROMPTS = (
    "A brown and white dog is running on the grass",
    "A cup of coffee on a wooden table beside a window",
)


def run(
    model: str, config_path: Path, out: Path, mode: str, steps: int, repeats: int
) -> None:
    config = yaml.safe_load(config_path.read_text())
    for stage in config["stages"]:
        if mode == "local":
            del stage["kv_transfer_config"]
        else:
            stage["worker_extension_cls"] = (
                "nixl_e2e_metrics.NixlMetricsWorkerExtension"
            )
            if mode == "ordinary":
                stage["kv_transfer_config"]["kv_connector_module_path"] = (
                    "nixl_page_reference"
                )
    run_config = out / "deploy.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=False))
    tokenizer, sequence_template = load_prompt_builder(model)
    started = time.perf_counter()
    omni = Omni(
        model=model,
        deploy_config=str(run_config),
        trust_remote_code=True,
        init_timeout=1800,
        stage_init_timeout=1800,
    )
    initialization_s = time.perf_counter() - started
    records = []
    for iteration in range(-1, repeats):
        for prompt_index, text in enumerate(PROMPTS):
            prompt_tokens = build_ar_tokens(tokenizer, text, sequence_template)
            sampling = [
                clone_sampling_params(params)
                for params in omni.default_sampling_params_list
            ]
            for params in sampling:
                if isinstance(params, OmniDiffusionSamplingParams):
                    params.seed = 1234 + prompt_index
                    params.num_inference_steps = steps
                    params.guidance_scale = 2.0
                    params.height = params.width = 512
                    params.num_outputs_per_prompt = 1
                elif isinstance(params, SamplingParams):
                    params.seed = 1234 + prompt_index
                    params.stop_token_ids = resolve_stop_token_ids(
                        task="t2i",
                        bot_task="think",
                        tokenizer=tokenizer.tokenizer,
                        image_size="512x512",
                    )
            start = time.perf_counter()
            outputs = list(
                omni.generate(
                    {
                        "prompt": text,
                        "prompt_token_ids": prompt_tokens,
                        "use_system_prompt": "None",
                        "modalities": ["image"],
                        "height": 512,
                        "width": 512,
                    },
                    sampling_params_list=sampling,
                )
            )
            elapsed = time.perf_counter() - start
            images = extract_images_from_outputs(outputs)
            assert len(images) == 1 and isinstance(images[0], Image.Image)
            image = images[0].convert("RGB")
            assert image.size == (512, 512) and np.asarray(image).std() > 1
            filename = f"prompt-{prompt_index}-repeat-{iteration}.png"
            image.save(out / filename)
            records.append(
                {
                    "prompt": text,
                    "prompt_index": prompt_index,
                    "repeat": iteration,
                    "seed": 1234 + prompt_index,
                    "steps": steps,
                    "elapsed_s": elapsed,
                    "image": filename,
                    "image_sha256": hashlib.sha256(
                        (out / filename).read_bytes()
                    ).hexdigest(),
                    "stage_metrics": [output.metrics for output in outputs],
                    "kv_transfer_params": [
                        output.kv_transfer_params
                        for output in outputs
                        if output.kv_transfer_params is not None
                    ],
                }
            )
            (out / "requests.json").write_text(json.dumps(records, indent=2) + "\n")
            print(
                f"{mode} prompt={prompt_index} repeat={iteration} elapsed={elapsed:.3f}s",
                flush=True,
            )
    metrics = None
    if mode != "local":
        deadline = time.monotonic() + 30
        while True:
            metrics = omni.engine.collective_rpc("nixl_page_metrics")
            flat = [worker for stage in metrics for worker in stage]
            if all(
                worker["active_reads"]
                == worker["source_leases"]
                == worker["destination_reservations"]
                == 0
                for worker in flat
            ):
                break
            assert time.monotonic() < deadline, (
                "Native leases or reservations did not drain; processes retained"
            )
            time.sleep(0.1)
        assert all(worker["page_pool_registrations"] == 1 for worker in flat)
        if mode == "pages":
            assert sum(worker["page_reads"] for worker in flat) == 2 * len(records)
            assert sum(worker["page_exports"] for worker in flat) == len(records)
            assert all(worker["gets"] == 0 and worker["errors"] == 0 for worker in flat)
            published = sum(worker["page_bytes_published"] for worker in flat)
            completed = sum(worker["page_bytes_completed"] for worker in flat)
            assert published > 0 and completed == 2 * published
        else:
            assert sum(worker["gets"] for worker in flat) == 2 * len(records)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "model": model,
                "sequence_template": sequence_template,
                "initialization_s": initialization_s,
                "request_count": len(records),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    omni.close()
    print(f"{mode}: close returned; inspect Worker PIDs for natural exit", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "vllm_omni/deploy/hunyuan-image-3-nixl-pages.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "ordinary", "pages"), required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.model, args.config, args.out, args.mode, args.steps, args.repeats)


if __name__ == "__main__":
    main()
