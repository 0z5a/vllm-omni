"""Finite, task-peer-only SOCKS relay over existing SSH access."""

import argparse
import ipaddress
import select
import socket
import socketserver
import struct
import subprocess
import time

THOR = "192.168.1.97"
WSL = "172.25.93.38"


def receive(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("SOCKS request closed")
        data.extend(chunk)
    return bytes(data)


class Relay(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        version, count = receive(self.request, 2)
        assert version == 5 and 0 in receive(self.request, count)
        self.request.sendall(b"\x05\x00")
        request = receive(self.request, 10)
        assert request[:4] == b"\x05\x01\x00\x01"
        host = str(ipaddress.IPv4Address(request[4:8]))
        port = struct.unpack("!H", request[8:10])[0]
        if host == THOR:
            command = ["ssh", "-S", "/tmp/codex-thor-a3-iteration-20261001c.sock", "-p", "2223", "jwipc@127.0.0.1"]
            bridge = f"/usr/bin/python3 /home/jwipc/0z5a/work/nixl-hunyuan-20261001/socket_stdio.py {host} {port}"
        else:
            assert host == WSL
            command = ["ssh", "-i", "/Users/0z5a/.ssh/id_ed25519_winpc", "-oHostKeyAlias=100.121.185.112", "Administrator@192.168.10.6"]
            bridge = f"wsl -d Ubuntu-24.04 -u root -- /usr/bin/python3 /mnt/d/0z5a/nixl-thor5080-20261001/socket_stdio.py {host} {port}"
        child = subprocess.Popen(command + ["-oBatchMode=yes", bridge], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert child.stdin is not None and child.stdout is not None
        self.request.sendall(b"\x05\x00\x00\x01" + bytes(6))
        while True:
            ready, _, _ = select.select([self.request, child.stdout], [], [])
            if self.request in ready:
                data = self.request.recv(256 * 1024)
                if not data:
                    break
                child.stdin.write(data)
                child.stdin.flush()
            if child.stdout in ready:
                data = child.stdout.read1(256 * 1024)
                if not data:
                    break
                self.request.sendall(data)
        child.stdin.close()
        while child.stdout.read(256 * 1024):
            pass
        child.stdout.close()
        child.wait()
        print(host, port, "closed", child.returncode, flush=True)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def verify_request(self, request: socket.socket, client_address: tuple[str, int]) -> bool:
        return client_address[0] in {"127.0.0.1", "192.168.10.6"}


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5591)
parser.add_argument("--seconds", type=int, required=True)
args = parser.parse_args()
with Server(("0.0.0.0", args.port), Relay) as server:
    server.timeout = 1
    deadline = time.monotonic() + args.seconds
    print("Relay ready", args.port, flush=True)
    while time.monotonic() < deadline:
        server.handle_request()
