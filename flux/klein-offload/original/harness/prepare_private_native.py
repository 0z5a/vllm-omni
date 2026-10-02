"""Extract verified compiler packages using task-private, signed Ubuntu indexes."""

import hashlib
import json
import os
import shlex
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "gcc-13",
    "gcc-13-x86-64-linux-gnu",
    "gcc-13-base",
    "cpp-13",
    "cpp-13-x86-64-linux-gnu",
    "g++-13",
    "g++-13-x86-64-linux-gnu",
    "libgcc-13-dev",
    "libstdc++-13-dev",
    "libpython3.12-dev",
    "libc6-dev",
    "linux-libc-dev",
    "libisl23",
    "libmpc3",
    "libmpfr6",
    "libcc1-0",
    "libnuma1",
    "libgmp10",
]


def main():
    env = os.environ.copy()
    env["APT_CONFIG"] = str(ROOT / "inputs/private-apt.conf")

    def apt(command):
        return subprocess.run(command, env=env, capture_output=True, text=True, check=True).stdout

    records = []
    for package in PACKAGES:
        policy = apt(["apt-cache", "policy", package])
        version = next(line.split(":", 1)[1].strip() for line in policy.splitlines() if "Candidate:" in line)
        metadata = apt(["apt-cache", "show", package + "=" + version])
        fields = dict(line.split(": ", 1) for line in metadata.splitlines() if ": " in line and not line.startswith(" "))
        records.append(
            {
                "package": package,
                "version": version,
                "filename": fields["Filename"],
                "sha256": fields["SHA256"],
                "bytes": int(fields["Size"]),
            }
        )
    uris = apt(["apt-get", "--print-uris", "download", *[r["package"] + "=" + r["version"] for r in records]])
    urls = {
        Path(unquote(urlsplit(parts[0]).path)).name: parts[0]
        for line in uris.splitlines()
        if line.startswith("'")
        for parts in [shlex.split(line)]
    }
    for item in records:
        item["url"] = urls[Path(item["filename"]).name]
    (ROOT / "evidence/alt-private-compiler-package-plan-v4.json").write_text(json.dumps(records, indent=2))
    cache = ROOT / "cache/compiler-debs"

    def fetch(item):
        path = cache / Path(item["filename"]).name
        if not path.exists():
            partial = path.with_suffix(".partial")
            with urllib.request.urlopen(item["url"], timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(2**20):
                    output.write(chunk)
            partial.rename(path)
        assert path.stat().st_size == item["bytes"], item["package"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == item["sha256"], item["package"]
        print(item["package"], item["version"], item["bytes"], "verified", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(fetch, records))
    native = ROOT / "dependencies/native"
    for path in paths:
        subprocess.run(["dpkg-deb", "-x", str(path), str(native)], check=True)
    (ROOT / "evidence/alt-private-compiler-v4.exit").write_text("0")
    print("Extracted", len(paths), "verified packages only into", native, flush=True)


if __name__ == "__main__":
    main()
