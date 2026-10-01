# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check FP8 AR loading and a forward pass before loading the full DiT."""

import argparse
import json
import time
from pathlib import Path

import yaml
from nixl_no_kill import install
from vllm import SamplingParams

install()

from nixl_hunyuan_prompt import build_ar_tokens, load_prompt_builder  # noqa: E402

from vllm_omni import Omni  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage = yaml.safe_load(args.config.read_text())["stages"][0]
    del stage["kv_transfer_config"]
    config = {"pipeline": "hunyuan_image3_ar", "async_chunk": False, "stages": [stage]}
    path = args.out / "deploy.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    tokenizer, sequence_template = load_prompt_builder(args.model)
    prompt_tokens = build_ar_tokens(tokenizer, "A brown and white dog is running on the grass", sequence_template)
    started = time.perf_counter()
    omni = Omni(model=args.model, deploy_config=str(path), trust_remote_code=True, init_timeout=1800)
    outputs = list(
        omni.generate(
            {"prompt_token_ids": prompt_tokens, "modalities": ["text"]},
            sampling_params_list=[SamplingParams(temperature=0, max_tokens=8, seed=1234)],
        )
    )
    assert len(outputs) == 1
    (args.out / "result.json").write_text(
        json.dumps({"elapsed_s": time.perf_counter() - started, "outputs": str(outputs)}, indent=2) + "\n"
    )
    omni.close()
    print("AR checkpoint load and forward pass completed with natural shutdown", flush=True)


if __name__ == "__main__":
    main()
