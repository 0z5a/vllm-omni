"""Deliver the remote DiT's normal shutdown message before closing its socket."""

import zmq

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

ORIGINAL_SHUTDOWN = StageDiffusionClient.shutdown


def remote_shutdown(client: StageDiffusionClient) -> None:
    if client._proc_manager is not None:
        ORIGINAL_SHUTDOWN(client)
        return
    client._shutting_down = True
    client._request_socket.setsockopt(zmq.SNDTIMEO, 10000)
    client._request_socket.setsockopt(zmq.LINGER, 10000)
    client._request_socket.send(client._encoder.encode({"type": "shutdown"}))
    client._request_socket.close()
    client._response_socket.close(linger=0)
    client._zmq_ctx.term()


def install() -> None:
    StageDiffusionClient.shutdown = remote_shutdown
