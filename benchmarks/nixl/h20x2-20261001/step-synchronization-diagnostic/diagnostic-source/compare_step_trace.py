"""Find the first full-tensor divergence in the finite ordinary diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = args.directory
trace = root / "trace"


def load(request, phase):
    return torch.load(
        trace / f"request-{request}-{phase}.pt", weights_only=True, map_location="cpu"
    )


def compare(left, right):
    delta = left.float() - right.float()
    return {
        "equal": bool(torch.equal(left, right)),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "rmse": float(delta.square().mean().sqrt()),
        "max": float(delta.abs().max()),
    }


reports = []
for prompt in range(2):
    steps = []
    for step in range(50):
        a, b = (
            load(prompt, f"prediction-{step:02d}"),
            load(prompt + 2, f"prediction-{step:02d}"),
        )
        sa, sb = (
            load(prompt, f"scheduler-{step:02d}"),
            load(prompt + 2, f"scheduler-{step:02d}"),
        )
        steps.append(
            {
                "step": step,
                "prediction": compare(a["prediction"], b["prediction"]),
                "scheduler": {
                    key: compare(sa[key], sb[key])
                    for key in ("input", "prediction", "output", "sigmas")
                },
                "timestep_equal": bool(torch.equal(sa["timestep"], sb["timestep"])),
                "step_index_equal": sa["step_index_after"] == sb["step_index_after"],
            }
        )
    va, vb = load(prompt, "vae"), load(prompt + 2, "vae")
    reports.append(
        {
            "prompt_index": prompt,
            "first_prediction_divergence": next(
                (row["step"] for row in steps if not row["prediction"]["equal"]), None
            ),
            "steps": steps,
            "vae": {key: compare(va[key], vb[key]) for key in ("input", "output")},
        }
    )
image_checks = []
records = json.loads((root / "requests.json").read_text())
for request, record in enumerate(records):
    vae = load(request, "vae")
    pipeline = load(request, "pipeline-output")["image"]
    expected = np.asarray(
        VaeImageProcessor().postprocess(
            pipeline, output_type="pil", do_denormalize=[True]
        )[0]
    )
    actual = np.asarray(Image.open(root / record["image"]).convert("RGB"))
    image_checks.append(
        {
            "request": request,
            "vae_pipeline_equal": bool(torch.equal(vae["output"].squeeze(2), pipeline)),
            "pipeline_png_equal": bool(np.array_equal(expected, actual)),
            "latent_finite": bool(torch.isfinite(vae["input"]).all()),
            "image_finite": bool(torch.isfinite(pipeline).all()),
            "image_range": [float(pipeline.min()), float(pipeline.max())],
        }
    )
attention = [
    json.loads(line) for line in (trace / "attention.jsonl").read_text().splitlines()
]
attention_by_key = {
    (row["request"], row["step"], row["layer"]): row for row in attention
}
attention_checks = []
for prompt in range(2):
    for step in [None, *range(50)]:
        for layer in (0, 31):
            a = attention_by_key[(prompt, step, layer)]
            b = attention_by_key[(prompt + 2, step, layer)]
            attention_checks.append(
                {
                    "prompt_index": prompt,
                    "step": step,
                    "layer": layer,
                    "equal": {
                        key: a[key]["sha256"] == b[key]["sha256"]
                        for key in ("query", "key", "value")
                    },
                }
            )
result = {
    "repeats": reports,
    "image_checks": image_checks,
    "attention_checks": attention_checks,
}
(root / "step-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
for row in reports:
    print(
        "prompt",
        row["prompt_index"],
        "first_prediction_divergence",
        row["first_prediction_divergence"],
        "vae",
        row["vae"],
    )
print("image_checks", image_checks)
