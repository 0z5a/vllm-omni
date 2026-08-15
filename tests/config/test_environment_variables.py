# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drift checks for the reviewed environment-variable inventory."""

import ast
import builtins
from collections import Counter
from pathlib import Path

import pytest

from vllm_omni.config.environment_variables import (
    ENVIRONMENT_VARIABLES,
    EnvironmentVariableCategory,
    ModelEnvironmentVariableDisposition,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "vllm_omni"
_REFERENCE_PAGE = _REPO_ROOT / "docs" / "configuration" / "environment_variables.md"


def _literal_environment_accesses(path: Path) -> set[str]:
    """Return uppercase literal names directly accessed through ``os``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()

    for node in ast.walk(tree):
        candidate: ast.expr | None = None
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "os" and func.attr == "getenv":
                    candidate = node.args[0]
                elif (
                    isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "os"
                    and func.value.attr == "environ"
                    and func.attr in {"get", "pop", "setdefault"}
                ):
                    candidate = node.args[0]
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            candidate = node.slice

        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str) and candidate.value.isupper():
            names.add(candidate.value)

    return names


def test_inventory_matches_reviewed_snapshot_counts():
    """Make an inventory expansion an explicit review decision."""
    category_counts = Counter(item.category for item in ENVIRONMENT_VARIABLES.values())
    assert category_counts == {
        EnvironmentVariableCategory.PUBLIC_OMNI: 22,
        EnvironmentVariableCategory.INHERITED_VLLM: 20,
        EnvironmentVariableCategory.PLATFORM_EXTERNAL: 26,
        EnvironmentVariableCategory.MODEL_SPECIFIC: 54,
        EnvironmentVariableCategory.BENCHMARK_TRANSITIONAL: 20,
        EnvironmentVariableCategory.INTERNAL: 2,
    }

    disposition_counts = Counter(
        item.model_disposition
        for item in ENVIRONMENT_VARIABLES.values()
        if item.category is EnvironmentVariableCategory.MODEL_SPECIFIC
    )
    assert {disposition: disposition_counts[disposition] for disposition in ModelEnvironmentVariableDisposition} == {
        ModelEnvironmentVariableDisposition.PROMOTE: 32,
        ModelEnvironmentVariableDisposition.REQUEST_SCOPE: 6,
        ModelEnvironmentVariableDisposition.EXTERNAL: 0,
        ModelEnvironmentVariableDisposition.INTERNALIZE: 11,
        ModelEnvironmentVariableDisposition.DEPRECATE_REMOVE: 5,
    }


def test_direct_literal_environment_accesses_are_classified():
    discovered: set[str] = set()
    for path in _PACKAGE_ROOT.rglob("*.py"):
        discovered.update(_literal_environment_accesses(path))

    assert discovered - ENVIRONMENT_VARIABLES.keys() == set()


def test_public_omni_variables_are_in_the_reference_page():
    reference = _REFERENCE_PAGE.read_text(encoding="utf-8")
    missing = {
        name for name, item in ENVIRONMENT_VARIABLES.items() if item.is_public_omni and f"`{name}`" not in reference
    }
    assert missing == set()


def test_secret_values_are_marked_for_redaction():
    assert {name for name, item in ENVIRONMENT_VARIABLES.items() if item.redact_value} == {
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "OPENAI_API_KEY",
    }


def test_collect_env_reports_only_safe_public_omni_values(monkeypatch):
    from collect_env import get_env_vars

    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "tea_cache")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("VLLM_OMNI_REPLICA_ID", "internal-replica")

    report = get_env_vars()

    assert "DIFFUSION_CACHE_BACKEND=tea_cache" in report.splitlines()
    assert "hf-secret" not in report
    assert "openai-secret" not in report
    assert "VLLM_OMNI_REPLICA_ID" not in report


def test_collect_env_survives_inventory_import_error(monkeypatch):
    from collect_env import get_env_vars

    real_import = builtins.__import__

    def fail_inventory_import(name, *args, **kwargs):
        if name == "vllm_omni.config.environment_variables":
            raise ImportError("broken vllm-omni installation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_inventory_import)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert "CUDA_VISIBLE_DEVICES=0" in get_env_vars().splitlines()
