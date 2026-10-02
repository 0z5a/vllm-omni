"""Check native imports and an actual Triton compilation in the private runtime."""

import argparse
import importlib
import importlib.metadata
import json
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(ROOT / "dependencies"))

import torch
import triton
import triton.language as tl


@triton.jit
def add_two(source, output, BLOCK: tl.constexpr):
    indices = tl.arange(0, BLOCK)
    tl.store(output + indices, tl.load(source + indices) + 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    modules = [
        "vllm._C_stable_libtorch",
        "transformers.models.qwen3.modeling_qwen3",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.diffusion.models.flux2_klein.quantization",
    ]
    loaded = {}
    for name in modules:
        loaded[name] = importlib.import_module(name).__file__
        print("Imported", name, flush=True)
    assert torch.cuda.is_available()
    source = torch.arange(1024, device="cuda", dtype=torch.float32)
    output = torch.empty_like(source)
    add_two[(1,)](source, output, BLOCK=1024)
    torch.testing.assert_close(output, source + 2, atol=0, rtol=0)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "vllm", "transformers", "diffusers", "triton", "flashinfer-python"]
        },
        "modules": loaded,
        "optional_audio": "Unused incompatible package preserved outside the image-runtime import path.",
        "triton_compiled_and_executed": True,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
