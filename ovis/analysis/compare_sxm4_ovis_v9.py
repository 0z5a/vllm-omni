"""Compare full-matrix Ovis images after the raw archive has been verified."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    assert (root / "evidence/a100-ovis-full-v9/exit").read_text().strip() == "0"
    hashes = json.loads((root / "sha256-index.json").read_text())
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    )
    plan = json.loads((root / "inputs/ovis-full-queue-v9.json").read_text())
    assert len(plan["steps"]) == 24
    images, names = {}, {}
    for step in plan["steps"]:
        argv = step["argv"]
        directory = root / "ovis/evidence" / Path(argv[argv.index("--output") + 1]).name
        settings = json.loads((directory / "config.json").read_text())["arguments"]
        resolution = (settings["height"], settings["width"])
        variant, mode = settings["variant"], settings["hsdp"]
        records = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        assert len(records) == 8 and (directory / "completed").is_file()
        repeats = {key[3] for key in images if key[:3] == (resolution, variant, mode)}
        repeat = len(repeats) + 1
        assert repeat in (1, 2)
        for record in records:
            image = directory / record["image"]
            assert (
                hashlib.sha256(image.read_bytes()).hexdigest() == record["image_sha256"]
            )
            if record["warmup"]:
                continue
            key = (
                resolution,
                variant,
                mode,
                repeat,
                record["prompt_index"],
                (record["index"] - 2) // 3,
            )
            with Image.open(image) as frame:
                images[key] = np.asarray(frame.convert("RGB"))
            names[key] = str(image.relative_to(root))
    assert len(images) == 144
    comparisons = []

    def compare(left, right, kind):
        a, b = images[left], images[right]
        assert a.shape == b.shape
        comparisons.append(
            {
                "left": names[left],
                "right": names[right],
                "kind": kind,
                "pixel_exact": bool(np.array_equal(a, b)),
                "mae_per_255": float(
                    np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
                ),
                "ssim": float(
                    structural_similarity(a, b, channel_axis=2, data_range=255)
                ),
            }
        )

    resolutions = sorted({key[0] for key in images})
    assert resolutions == [(512, 512), (768, 1024)]
    for resolution in resolutions:
        sheet = Image.new("RGB", (6 * 256, 6 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for variant_index, variant in enumerate(("bf16", "encoder")):
            for mode_index, mode in enumerate(("none", "block", "resident")):
                column = variant_index * 3 + mode_index
                for prompt in range(3):
                    compare(
                        (resolution, variant, mode, 1, prompt, 1),
                        (resolution, variant, mode, 2, prompt, 1),
                        "process_repeat",
                    )
                    for repeat in (1, 2):
                        key = (resolution, variant, mode, repeat, prompt, 1)
                        compare((*key[:-1], 0), key, "request_repeat")
                        if variant == "encoder":
                            compare(
                                (resolution, "bf16", mode, repeat, prompt, 1),
                                key,
                                "encoder_quantization",
                            )
                        if mode == "resident":
                            compare(
                                (resolution, variant, "block", repeat, prompt, 1),
                                key,
                                "resident_vs_block",
                            )
                        frame = Image.fromarray(images[key])
                        frame.thumbnail((256, 256))
                        x, y = column * 256, ((repeat - 1) * 3 + prompt) * 280
                        sheet.paste(frame, (x, y + 24))
                        draw.text(
                            (x + 4, y + 4),
                            f"{variant} {mode} r{repeat} p{prompt}",
                            fill="black",
                        )
        sheet.save(root / f"ovis-{resolution[0]}x{resolution[1]}-contact-sheet.png")
    assert len(comparisons) == 168
    hsdp_comparisons = []
    for resolution in resolutions:
        for variant in ("bf16", "encoder"):
            for mode in ("block", "resident"):
                for repeat in (1, 2):
                    for prompt in range(3):
                        compare(
                            (resolution, variant, "none", repeat, prompt, 1),
                            (resolution, variant, mode, repeat, prompt, 1),
                            "hsdp_vs_unsharded",
                        )
                        hsdp_comparisons.append(comparisons.pop())
    assert len(hsdp_comparisons) == 48
    (root / "ovis-full-v9-hsdp-data-comparisons.json").write_text(
        json.dumps(
            {
                "comparisons": hsdp_comparisons,
                "limits": "Same precision, prompt and seed across HSDP and unsharded runs. These existing-image comparisons supplement the NCCL data check; nonexact results require manual review and do not identify a transport failure on their own.",
            },
            indent=2,
        )
        + "\n"
    )
    (root / "ovis-full-v9-image-comparisons.json").write_text(
        json.dumps(
            {
                "comparisons": comparisons,
                "limits": "SSIM and MAE describe image differences. Subject counts, handles, instructions and fidelity still require manual review.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Verified all 192 image hashes; compared 144 timed images in {len(comparisons)} matched comparisons."
    )


if __name__ == "__main__":
    main()
