"""Run only the coordinated 1 MiB VRAM READ, with both leases held first."""

import json
import os
from pathlib import Path
import select
import shlex
import subprocess


root = Path(__file__).resolve().parents[1]
evidence = root / "evidence/thor5080/vram-peer-v2"
evidence.mkdir(exist_ok=True)
nodes = {
    "producer": ("/mnt/d/0z5a/nixl-thor5080-20261001", "GPU-99386c0c-e54e-e92e-9cf1-3cd91b6397f7", "8152cf95-168f-4e1e-8c12-ba6a61f1e0ab"),
    "consumer": ("/home/jwipc/0z5a/work/nixl-hunyuan-20261001", "GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600", "45fb4742-5aac-48ce-bcd7-0db6a26ee428"),
}
processes = {}
streams = {}
logs = {}
errors = {}
for role, (directory, uuid, boot) in nodes.items():
    env = {
        "CUDA_VISIBLE_DEVICES": "0", "UCX_TLS": "tcp,cuda_copy", "NIXL_TASK_ROOT": directory,
        "NIXL_EXPECT_GPU_UUID": uuid, "NIXL_EXPECT_BOOT_ID": boot,
        "LD_PRELOAD": directory + "/peer_socks_connect.so", "NIXL_TUNNEL_PORT": "5591",
    }
    if role == "producer":
        env.update(UCX_NET_DEVICES="eth0", NIXL_GPU_LOCK="/tmp/0z5a-sm120-perf.lock", NIXL_TUNNEL_PEER="192.168.1.97", NIXL_TUNNEL_PROXY="192.168.10.10")
        ssh = ["ssh", "-i", "/Users/0z5a/.ssh/id_ed25519_winpc", "-oHostKeyAlias=100.121.185.112", "-oBatchMode=yes", "Administrator@192.168.10.6"]
        command = ["/usr/bin/python3", directory + "/private_python.py"]
        prefix = "wsl -d Ubuntu-24.04 -u root -- "
    else:
        env.update(UCX_NET_DEVICES="p2p0", NIXL_GPU_LOCK="/tmp/codex-thor-perf.lock", NIXL_TUNNEL_PEER="172.25.93.38", NIXL_TUNNEL_PROXY="127.0.0.1", NIXL_QUEUE_STATUS="/home/jwipc/0z5a/work/verl-qwen-thor-20261001/queue.status")
        ssh = ["ssh", "-S", "/tmp/codex-thor-a3-iteration-20261001c.sock", "-p", "2223", "-oBatchMode=yes", "jwipc@127.0.0.1"]
        command = [directory + "/fast-runtime/bin/python"]
        prefix = ""
    output = directory + f"/native-vram-v2-{role}-proof.json"
    command += [directory + "/run_native_peer_gate.py", role, "172.25.93.38", output]
    remote = prefix + shlex.join(["/usr/bin/env", *[f"{key}={value}" for key, value in env.items()], *command])
    errors[role] = (evidence / f"{role}.stderr.log").open("w")
    logs[role] = (evidence / f"{role}.stdout.log").open("w")
    child = subprocess.Popen(ssh + [remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors[role])
    assert child.stdin is not None and child.stdout is not None
    processes[role] = child
    streams[child.stdout] = role
leases = set()
producer_started = consumer_started = False
buffers = {stream: b"" for stream in streams}
while streams:
    readable, _, _ = select.select(list(streams), [], [], 1)
    for stream in readable:
        chunk = os.read(stream.fileno(), 65536)
        role = streams[stream]
        if not chunk:
            del streams[stream]
            continue
        lines = (buffers[stream] + chunk).split(b"\n")
        buffers[stream] = lines.pop()
        for record in lines:
            line = record.decode()
            logs[role].write(line + "\n")
            logs[role].flush()
            print(role, line, flush=True)
            if line.startswith("LEASE_READY"):
                leases.add(role)
            if leases == set(nodes) and not producer_started and all(child.poll() is None for child in processes.values()):
                processes["producer"].stdin.write(b"GO\n")
                processes["producer"].stdin.flush()
                producer_started = True
            if role == "producer" and line.startswith("Native producer ready") and not consumer_started and processes["consumer"].poll() is None:
                processes["consumer"].stdin.write(b"GO\n")
                processes["consumer"].stdin.flush()
                consumer_started = True
results = {role: child.wait() for role, child in processes.items()}
for role in nodes:
    logs[role].close()
    errors[role].close()
(evidence / "exit-codes.json").write_text(json.dumps(results) + "\n")
print(results, flush=True)
assert producer_started and consumer_started and set(results.values()) == {0}
