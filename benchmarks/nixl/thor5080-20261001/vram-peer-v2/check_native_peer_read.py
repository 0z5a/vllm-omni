"""Verify native UCX DRAM READ across architectures before GPU deployment."""

import base64
import fcntl
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config


def receive(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        assert chunk, "Native peer control channel closed"
        data.extend(chunk)
    return bytes(data)


role, host, output = sys.argv[1:4]
gpu = "--gpu" in sys.argv
size = 1024 * 1024
expected = np.tile(np.arange(256, dtype=np.uint8), 4096)
array = expected if role == "producer" else np.zeros(size, dtype=np.uint8)
assert not torch.cuda.is_initialized()
if gpu:
    lease = (
        os.fdopen(int(os.environ["NIXL_GPU_LOCK_FD"]), "a", closefd=False)
        if "NIXL_GPU_LOCK_FD" in os.environ
        else open(os.environ["NIXL_GPU_LOCK"], "a")
    )
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease_started = float(os.environ.get("NIXL_GPU_LEASE_STARTED", time.perf_counter()))
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).strip()
    assert not active, active
    array = torch.from_numpy(array).to("cuda")
    torch.cuda.synchronize()
agent = nixl_agent(f"nixl-route-{role}", nixl_agent_config(backends=["UCX"], num_threads=1))
print("PHASE agent-ready", time.time(), flush=True)
pointer = array.data_ptr() if gpu else array.ctypes.data
memory_type = "VRAM" if gpu else "DRAM"
region = (pointer, size, 0, "")
registered = agent.register_memory([region], memory_type, backends=["UCX"])
print("PHASE registered", time.time(), flush=True)
assert torch.cuda.is_initialized() == gpu
if role == "producer":
    message = json.dumps({
        "metadata": base64.b64encode(agent.get_agent_metadata()).decode(),
        "pointer": pointer, "size": size,
    }).encode()
    with socket.socket() as listener:
        listener.settimeout(30)
        listener.bind((host, 5618))
        listener.listen(1)
        print("Native producer ready", flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(45)
            connection.sendall(struct.pack("!I", len(message)) + message)
            assert receive(connection, 4) == b"DONE"
else:
    with socket.create_connection((host, 5618), timeout=20) as connection:
        message = json.loads(receive(connection, struct.unpack("!I", receive(connection, 4))[0]))
        remote = agent.add_remote_agent(base64.b64decode(message["metadata"]))
        print("PHASE remote-added", time.time(), flush=True)
        local_list = agent.prep_xfer_dlist("NIXL_INIT_AGENT", [region[:3]], memory_type)
        remote_list = agent.prep_xfer_dlist(remote, [(message["pointer"], size, 0)], memory_type)
        transfer = agent.make_prepped_xfer("READ", local_list, [0], remote_list, [0])
        print("PHASE prepared", time.time(), flush=True)
        assert agent.transfer(transfer) in {"PROC", "DONE"}
        print("PHASE submitted", time.time(), flush=True)
        deadline = time.monotonic() + 30
        state = agent.check_xfer_state(transfer)
        while state == "PROC" and time.monotonic() < deadline:
            time.sleep(0.001)
            state = agent.check_xfer_state(transfer)
        assert state == "DONE", state
        actual = array.cpu().numpy() if gpu else array
        assert np.array_equal(actual, expected)
        agent.release_xfer_handle(transfer)
        agent.release_dlist_handle(local_list)
        agent.release_dlist_handle(remote_list)
        agent.remove_remote_agent(remote)
        connection.sendall(b"DONE")
agent.deregister_memory(registered)
assert torch.cuda.is_initialized() == gpu
actual = array.cpu().numpy() if gpu else array
proof = {"role": role, "bytes": size, "memory_type": memory_type,
         "sha256": hashlib.sha256(actual.tobytes()).hexdigest(), "cuda_initialized": gpu}
if gpu:
    proof.update(pid=os.getpid(), gpu_lease_elapsed_s=time.perf_counter() - lease_started)
Path(output).write_text(json.dumps(proof, indent=2) + "\n")
print(proof, flush=True)
