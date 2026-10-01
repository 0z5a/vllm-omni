"""Exercise normal shutdown delivery to a peer that connects after startup."""

import json
import os
import threading
import time
from pathlib import Path

import torch
import zmq
from vllm.utils import import_utils

import_utils.has_nixl_ep = lambda: False

from nixl_remote_shutdown import remote_shutdown
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder

client = object.__new__(StageDiffusionClient)
client._proc_manager = None
client._zmq_ctx = zmq.Context()
client._request_socket = client._zmq_ctx.socket(zmq.PUSH)
port = client._request_socket.bind_to_random_port("tcp://127.0.0.1")
client._response_socket = client._zmq_ctx.socket(zmq.PULL)
client._encoder = OmniMsgpackEncoder()
received = []


def peer() -> None:
    time.sleep(1)
    with zmq.Context() as context, context.socket(zmq.PULL) as receiver:
        receiver.setsockopt(zmq.RCVTIMEO, 15000)
        receiver.connect(f"tcp://127.0.0.1:{port}")
        received.append(OmniMsgpackDecoder().decode(receiver.recv()))


thread = threading.Thread(target=peer)
thread.start()
remote_shutdown(client)
thread.join(20)
assert not thread.is_alive() and received == [{"type": "shutdown"}]
assert client._zmq_ctx.closed and not torch.cuda.is_initialized()
root = Path(os.environ["NIXL_TASK_ROOT"])
(root / "remote-shutdown-proof.json").write_text(json.dumps({"received": received, "context_closed": True, "cuda_initialized": False}) + "\n")
print("Delayed peer received normal shutdown; contexts closed", flush=True)
