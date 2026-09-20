# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the MOSS-VL-Realtime contract checks without the repository's pytest plugins.

The repository test configuration requires plugins that are not present in every
validation environment (``pytest-asyncio`` among them). These checks only need
torch and the fixture files, so this runner executes the same functions
directly and prints one line per check, which keeps the evidence reproducible on
a bare host. The pytest files remain the source of truth for CI.

Example:
    PYTHONPATH=$PWD python tests/model_executor/models/moss_vl_realtime/run_checks.py
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

PARAMETRIZED_BY_CASE = {
    "test_fixture_hashes_match_manifest",
    "test_captured_contract_is_integer_exact",
    "test_visibility_expansion_recomputed_from_frames",
}
NEEDS_TMP_PATH = {
    "test_published_layout_is_accepted",
    "test_converted_checkpoint_is_refused",
    "test_wrong_architecture_is_refused",
    "test_unsharded_checkpoint_is_refused",
    "test_corrupt_config_is_refused",
    "test_missing_config_raises",
}
MODULES = (
    "test_native_contract",
    "test_attention_correctness",
    "test_reference_fixtures",
    "test_checkpoint_contract",
)


def call(module, name: str, function) -> None:
    if name in PARAMETRIZED_BY_CASE:
        for case in module.CASES:
            function(case)
        return
    if name in NEEDS_TMP_PATH:
        with tempfile.TemporaryDirectory() as tmp:
            function(Path(tmp))
        return
    if "prepare_module" in function.__code__.co_varnames[: function.__code__.co_argcount]:
        function(load_prepare_module())
        return
    function()


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_inputs", HERE.parents[2] / "assets" / "moss_vl_realtime" / "prepare_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sys.path.insert(0, str(HERE))
    failures: list[str] = []
    total = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in sorted(attribute for attribute in dir(module) if attribute.startswith("test_")):
            total += 1
            try:
                call(module, name, getattr(module, name))
                print(f"PASS {module_name}.{name}")
            except Exception as error:  # noqa: BLE001 - the runner reports, it does not handle
                failures.append(f"{module_name}.{name}: {type(error).__name__}: {error}")
                print(f"FAIL {module_name}.{name}: {type(error).__name__}: {error}")
    print(f"\n{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
