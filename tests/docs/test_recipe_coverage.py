# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT_DIR = Path(__file__).parents[2]
COVERAGE_PATH = ROOT_DIR / "docs/models/recipe_coverage.yaml"
SUPPORTED_MODELS_PATH = ROOT_DIR / "docs/models/supported_models.md"
VALID_STATUSES = {"published", "repository", "no_validated_recipe_yet"}
VALID_HARDWARE = {"nvidia", "amd", "ascend", "intel"}
HARDWARE_COLUMNS = {
    "nvidia": 4,
    "amd": 5,
    "ascend": 6,
    "intel": 7,
}


def load_entries() -> list[dict]:
    data = yaml.safe_load(COVERAGE_PATH.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert isinstance(data["entries"], list)
    return data["entries"]


def supported_model_ids() -> set[str]:
    model_ids = set()
    for line in SUPPORTED_MODELS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        columns = line.split("|")
        if len(columns) <= 3:
            continue
        model_ids.update(re.findall(r"`([^`]+/[^`]+)`", columns[3]))
    return model_ids


def supported_model_recipe_cells() -> dict[str, str]:
    recipe_cells = {}
    in_table = False
    for line in SUPPORTED_MODELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Architecture |"):
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            break
        if not in_table:
            continue

        columns = line.split("|")
        if len(columns) <= 8 or columns[1].strip().startswith("-"):
            continue
        for model_id in re.findall(r"`([^`]+/[^`]+)`", columns[3]):
            recipe_cells[model_id] = columns[8].strip()
    return recipe_cells


def expected_recipe_link(entry: dict) -> str:
    recipe = entry["recipe"]
    if recipe["status"] == "published":
        return recipe["published_url"]
    if recipe["status"] == "repository":
        return f"https://github.com/vllm-project/vllm-omni/blob/main/{recipe['repository_path']}"
    return "No validated recipe yet"


def supported_model_hardware_cells() -> dict[str, list[str]]:
    hardware_cells = {}
    in_table = False
    for line in SUPPORTED_MODELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Architecture |"):
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            break
        if not in_table:
            continue

        columns = line.split("|")
        if len(columns) <= 8 or columns[1].strip().startswith("-"):
            continue
        cells = [columns[index].strip() for index in HARDWARE_COLUMNS.values()]
        for model_id in re.findall(r"`([^`]+/[^`]+)`", columns[3]):
            hardware_cells[model_id] = cells
    return hardware_cells


def test_registry_keys_are_exact_supported_checkpoints():
    entries = load_entries()
    keys = [(entry["model"], entry["task"]) for entry in entries]

    assert len(keys) == len(set(keys))
    assert all(entry["model"] in supported_model_ids() for entry in entries)


def test_supported_models_render_registry_recipe_links():
    recipe_cells = supported_model_recipe_cells()

    for entry in load_entries():
        assert expected_recipe_link(entry) in recipe_cells[entry["model"]]


def test_registry_entries_have_explicit_recipe_precedence():
    for entry in load_entries():
        assert set(entry) >= {"model", "task", "examples", "recipe"}
        assert isinstance(entry["examples"], list)

        recipe = entry["recipe"]
        status = recipe["status"]
        assert status in VALID_STATUSES
        assert set(recipe.get("hardware", [])) <= VALID_HARDWARE

        if status == "published":
            assert recipe["published_url"].startswith("https://recipes.vllm.ai/")
        elif status == "repository":
            assert recipe["repository_path"].startswith("recipes/")
            assert "published_url" not in recipe
        else:
            assert "published_url" not in recipe
            assert "repository_path" not in recipe

        paths = [*entry["examples"], recipe.get("repository_path")]
        for relative_path in filter(None, paths):
            path = Path(relative_path)
            assert not path.is_absolute()
            assert (ROOT_DIR / path).is_file(), relative_path


def test_supported_models_render_recipe_hardware():
    hardware_cells = supported_model_hardware_cells()

    for entry in load_entries():
        recipe_hardware = set(entry["recipe"].get("hardware", []))
        cells = hardware_cells[entry["model"]]
        for hardware, cell in zip(HARDWARE_COLUMNS, cells, strict=True):
            assert (cell == "✅︎") == (hardware in recipe_hardware)
