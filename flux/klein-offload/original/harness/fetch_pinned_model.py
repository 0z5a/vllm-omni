"""Download one fixed model into the task and verify official file hashes."""

import argparse
import hashlib
import json
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["FLUX.2-klein-4B", "FLUX.1-Kontext-dev", "Ovis-Image-7B"])
    args = parser.parse_args()
    site.addsitedir(str(ROOT / "dependencies/download"))
    from huggingface_hub import snapshot_download

    manifest = json.loads((ROOT / "evidence" / f"{args.model}-hub-manifest.json").read_text())
    files = [
        item
        for item in manifest["files"]
        if item["path"] == "model_index.json"
        or item["path"].startswith(
            ("scheduler/", "text_encoder/", "text_encoder_2/", "tokenizer/", "tokenizer_2/", "transformer/", "vae/")
        )
    ]
    assert files, "The model manifest contains no pipeline components"
    target = ROOT / "models" / args.model
    assert not (target / ".verified-complete").exists(), "This fixed model is already complete"
    snapshot_download(
        manifest["repo_id"],
        revision=manifest["revision"],
        local_dir=target,
        allow_patterns=[item["path"] for item in files],
        max_workers=3,
    )
    verified = []
    for item in files:
        path = target / item["path"]
        assert path.stat().st_size == item["size"], item["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if item["sha256"] is not None:
            assert digest == item["sha256"], item["path"]
        verified.append({"path": item["path"], "size": item["size"], "sha256": digest})
        print(item["path"], item["size"], "verified", flush=True)
    result = {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "files": verified}
    (target / ".verified-complete").write_text(json.dumps(result, indent=2))
    print(args.model, len(files), "official files complete", flush=True)


if __name__ == "__main__":
    main()
