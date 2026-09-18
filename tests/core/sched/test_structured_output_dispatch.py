# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dispatch for advancing a request's grammar with just-sampled tokens.

``accept_structured_output_tokens`` must work against both the released
``should_advance`` + per-request ``grammar`` layout and the newer
``StructuredOutputManager.accept_tokens`` layout.

Two properties are load-bearing and pinned here:

* it is a module-level function, so scheduler stubs that call
  ``update_from_output`` unbound do not have to grow an attribute for it;
* it dispatches on the manager *class*, because an ``getattr`` on a ``MagicMock``
  instance is never ``None`` and would make the older branch unreachable.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import (
    OmniSchedulerMixin,
    accept_structured_output_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request():
    grammar = MagicMock()
    grammar.accept_tokens.return_value = True
    return SimpleNamespace(
        request_id="req-0",
        structured_output_request=SimpleNamespace(grammar=grammar),
    )


class _NewLayoutManager:
    """Manager whose class advertises ``accept_tokens``."""

    def __init__(self, accepted: bool = True):
        self.accepted = accepted
        self.calls: list[tuple[str, list[int]]] = []

    def accept_tokens(self, request, token_ids) -> bool:
        self.calls.append((request.request_id, list(token_ids)))
        return self.accepted

    def should_advance(self, request) -> bool:  # pragma: no cover - must not be used
        raise AssertionError("the newer layout must not gate on should_advance")


def test_helper_is_module_level_not_a_mixin_method():
    # A mixin method disappears when tests call ``update_from_output`` unbound
    # on a stub namespace; a module-level helper is reachable from both callers.
    assert not hasattr(OmniSchedulerMixin, "_accept_structured_output_tokens")
    assert callable(accept_structured_output_tokens)


def test_manager_without_accept_tokens_gates_on_should_advance():
    request = _request()
    manager = MagicMock()
    manager.should_advance.return_value = False

    assert accept_structured_output_tokens(manager, request, [7]) is True
    request.structured_output_request.grammar.accept_tokens.assert_not_called()


def test_manager_without_accept_tokens_accepts_through_the_grammar():
    request = _request()
    manager = MagicMock()
    manager.should_advance.return_value = True

    assert accept_structured_output_tokens(manager, request, [7, 8]) is True
    request.structured_output_request.grammar.accept_tokens.assert_called_once_with("req-0", [7, 8])


def test_manager_without_accept_tokens_reports_a_grammar_rejection():
    request = _request()
    request.structured_output_request.grammar.accept_tokens.return_value = False
    manager = MagicMock()
    manager.should_advance.return_value = True

    assert accept_structured_output_tokens(manager, request, [7]) is False


def test_manager_class_with_accept_tokens_is_preferred():
    request = _request()
    manager = _NewLayoutManager()

    assert accept_structured_output_tokens(manager, request, [1, 2]) is True
    assert manager.calls == [("req-0", [1, 2])]
    request.structured_output_request.grammar.accept_tokens.assert_not_called()


def test_manager_class_with_accept_tokens_reports_rejection():
    request = _request()
    manager = _NewLayoutManager(accepted=False)

    assert accept_structured_output_tokens(manager, request, [3]) is False
