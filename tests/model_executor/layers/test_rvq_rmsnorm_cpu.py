# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CPU tensor metadata checks for the prototype's pointer-dispatch guard.

The guard expression is extracted without importing Triton or the kernel
module. Every tensor is explicitly on CPU; no CUDA method is called.
"""

import ast
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from types import CodeType, ModuleType
from typing import ClassVar

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class StorageGuardTests(unittest.TestCase):
    torch: ClassVar[ModuleType]
    storage_expression: ClassVar[CodeType]
    lazy_expression: ClassVar[CodeType]
    epsilon_predicate: ClassVar[Callable]

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("optional CPU Torch installation required") from error
        cls.torch = torch
        tree = ast.parse((Path(__file__).resolve().parents[3] / "benchmarks/kernels/rvq_rmsnorm.py").read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inspect_inputs"
        )
        guards = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "all"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
            and isinstance(node.args[0].generators[0].iter, ast.Name)
            and node.args[0].generators[0].iter.id == "tensors"
        ]
        assert len(guards) == 1
        cls.storage_expression = compile(ast.Expression(guards[0]), "prototype-storage-guard", "eval")
        cls.lazy_expression = compile(
            ast.Expression(next(node for node in function.body if isinstance(node, ast.If)).test),
            "lazy-view-guard",
            "eval",
        )
        epsilon_function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "supported_epsilon"
        )
        namespace: dict = {}
        exec(
            compile(ast.Module(body=[epsilon_function], type_ignores=[]), "prototype-epsilon-guard", "exec"), namespace
        )
        cls.epsilon_predicate = staticmethod(namespace["supported_epsilon"])

    def accepted(self, tensors):
        namespace = {"torch": self.torch, "tensors": tensors}
        return not eval(self.lazy_expression, namespace) and eval(self.storage_expression, namespace)

    def inputs(self):
        torch = self.torch
        return [
            torch.zeros((1, 16, 1), dtype=torch.int64, device="cpu"),
            torch.ones((32, 4), dtype=torch.float16, device="cpu"),
            torch.zeros((1, 16, 1), dtype=torch.int64, device="cpu"),
            torch.ones((4,), dtype=torch.float16, device="cpu"),
        ]

    def test_dense_parameter_and_broadcast_zero_stride_remain_supported_storage(self):
        torch = self.torch
        tensors = self.inputs()
        self.assertTrue(self.accepted(tensors))
        tensors[3] = torch.nn.Parameter(tensors[3])
        self.assertTrue(self.accepted(tensors))
        tensors[3] = torch.ones((1,), device="cpu", dtype=torch.float16).expand(4)
        self.assertEqual(tensors[3].stride(), (0,))
        self.assertTrue(self.accepted(tensors))

    def test_lazy_negative_view_on_each_input_requires_native_dispatch(self):
        torch = self.torch
        for index in range(4):
            with self.subTest(index=index):
                tensors = self.inputs()
                original = tensors[index]
                tensors[index] = torch._neg_view(original)
                self.assertEqual(type(tensors[index]), torch.Tensor)
                self.assertEqual(tensors[index].data_ptr(), original.data_ptr())
                self.assertTrue(tensors[index].is_neg())
                self.assertFalse(self.accepted(tensors))

    def test_sparse_nested_and_tensor_subclasses_require_native_dispatch(self):
        torch = self.torch
        tensors = self.inputs()
        tensors[3] = torch.sparse_coo_tensor(
            [[0, 2]], [1.0, 2.0], (4,), dtype=torch.float16, device="cpu", check_invariants=True
        )
        self.assertFalse(self.accepted(tensors))
        tensors = self.inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensors[3] = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)], device="cpu")
        self.assertTrue(tensors[3].is_nested)
        self.assertFalse(self.accepted(tensors))

        from torch import Tensor

        class TensorSubclass(Tensor):
            pass

        tensors = self.inputs()
        tensors[3] = tensors[3].as_subclass(TensorSubclass)
        self.assertFalse(self.accepted(tensors))

    def test_large_python_integer_epsilon_preserves_native_scalar_error(self):
        torch = self.torch
        variance = torch.ones((1,), device="cpu", dtype=torch.float32)
        self.assertFalse(self.epsilon_predicate(10**30))
        with self.assertRaises(OverflowError):
            variance + 10**30
        self.assertTrue(self.epsilon_predicate(float(10**30)))
        self.assertTrue(torch.isfinite(variance + float(10**30)).all().item())
        rounding_counterexample = 2**53 + 2**29 + 1
        self.assertNotEqual(
            (variance + rounding_counterexample).item(), (variance + float(rounding_counterexample)).item()
        )
        self.assertFalse(self.epsilon_predicate(rounding_counterexample))
        for epsilon in (0, 2**53, 0.0, 1e-5):
            self.assertTrue(self.epsilon_predicate(epsilon))
        for epsilon in (True, -1, 2**53 + 1, 2**63 - 1, 2**63, 10**400, float("nan"), float("inf")):
            self.assertFalse(self.epsilon_predicate(epsilon))


if __name__ == "__main__":
    unittest.main()
