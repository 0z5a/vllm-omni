"""Carry one peer TCP connection over an ordinary SSH command channel."""

import os
import select
import socket
import sys

host, port = sys.argv[1:]
assert host in {"192.168.1.97", "172.25.93.38"}
with socket.create_connection((host, int(port)), timeout=20) as connection:
    connection.settimeout(None)
    readers = [0, connection]
    while readers:
        ready, _, _ = select.select(readers, [], [])
        if 0 in ready:
            data = os.read(0, 256 * 1024)
            if data:
                connection.sendall(data)
            else:
                connection.shutdown(socket.SHUT_WR)
                readers.remove(0)
        if connection in ready:
            data = connection.recv(256 * 1024)
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
