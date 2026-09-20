"""Check every trained adapter dimension against real base-checkpoint headers."""

import json
from pathlib import Path

from safetensors import safe_open

base = Path("/home/kxqandccx/omni-1217-20260919/models/qwen-image-edit-2509/transformer")
shapes = {}
for shard in sorted(base.glob("*.safetensors")):
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            assert key not in shapes, key
            shapes[key] = list(tensors.get_slice(key).get_shape())
root = Path("/home/kxqandccx/omni-1217-20260919/models/lora")
reports = []
for name in ("edit-r1", "cocoedit"):
    adapter = json.loads((root / name / "verification.json").read_text())
    targets = set()
    for key, shape in adapter["tensor_shapes"].items():
        module = key.removeprefix("transformer.").rsplit(".lora_", 1)[0]
        base_shape = shapes[module + ".weight"]
        assert len(base_shape) == 2 and len(shape) == 2
        if key.endswith(".lora_A.weight"):
            assert shape == [adapter["config"]["r"], base_shape[1]], (key, shape, base_shape)
        else:
            assert key.endswith(".lora_B.weight")
            assert shape == [base_shape[0], adapter["config"]["r"]], (key, shape, base_shape)
        targets.add(module)
    assert len(targets) == 480
    report = {
        "adapter": name,
        "logical_targets": len(targets),
        "tensor_shapes_checked": len(adapter["tensor_shapes"]),
        "all_checkpoint_targets_exist": True,
        "all_shapes_match": True,
    }
    reports.append(report)
    print(json.dumps(report), flush=True)
(root / "qwen-pair-target-report.json").write_text(json.dumps(reports, indent=2) + "\n")
