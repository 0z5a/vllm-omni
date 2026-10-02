"""Record the unchanged CPU copy and GPU usage during a finite Thor queue."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copy-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.command
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def observe():
        with output.open("x") as log:
            while True:
                gpu = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                copy = subprocess.run(
                    ["cat", f"/proc/{args.copy_pid}/io"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "gpu": gpu.stdout,
                            "gpu_rc": gpu.returncode,
                            "copy_io": copy.stdout,
                            "copy_io_rc": copy.returncode,
                        }
                    )
                    + "\n"
                )
                log.flush()
                if stop.wait(10):
                    return

    monitor = threading.Thread(target=observe)
    monitor.start()
    try:
        result = subprocess.run(args.command, check=False).returncode
    finally:
        stop.set()
        monitor.join()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
