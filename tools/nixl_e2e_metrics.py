# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker RPC used to prove real model requests took the expected page path."""

from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

from vllm_omni.distributed.omni_connectors.connectors.nixl_kv_connector import OmniNixlKVConnector


class NixlMetricsWorkerExtension:
    def nixl_page_metrics(self) -> dict[str, int]:
        connector = get_kv_transfer_group()
        if not isinstance(connector, OmniNixlKVConnector):
            raise TypeError("E2E page metrics require the native Omni NIXL bridge")
        return connector.page_metrics()
