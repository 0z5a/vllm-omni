#!/home/gongji/0z5a/bin/python
# SPDX-License-Identifier: Apache-2.0
"""Offline text-to-image benchmark harness for Qwen-Image-2.1 (vllm-omni).

Runs exactly ONE text-to-image generation through vllm-omni's offline ``Omni``
entry point and prints a single machine-readable JSON object on stdout.

Design notes (all verified against the repo checkout, see README.md):
  * ``Omni`` is the offline entry point (``vllm_omni.entrypoints.omni.Omni``),
    driven with ``omni.generate(prompt_dict, sampling_params_list=[...])``.
    This mirrors ``examples/offline_inference/text_to_image/text_to_image.py``.
  * The diffusion stage runs in a *spawned subprocess*
    (``StageConfigFactory.create_default_diffusion`` sets ``runtime.process=True``),
    therefore in-process ``torch.cuda`` peaks do NOT cover the model work.
    We report them anyway (required), plus the worker-reported peak, plus an
    external device-wide memory poll.
  * fd 1 is redirected to fd 2 right after startup so that vLLM/vllm-omni log
    noise (including from the spawned stage process, which inherits fd 1) never
    pollutes stdout. The final JSON is written to the *saved* original stdout fd.

Anything that cannot be imported without weights (torch/vllm/vllm_omni) is
imported lazily so that ``--help``, ``--dry-run``, ``--preflight-only`` and
``--check-imports`` always work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# This harness must never send process termination signals, including on startup failure.
from multiprocessing.process import BaseProcess

_original_kill = os.kill


def signal_guard(pid: int, sig: int) -> None:
    if sig:
        raise RuntimeError(f"Process signals are prohibited: pid={pid}, signal={sig}")
    _original_kill(pid, sig)


def prohibit_process_signal(self) -> None:
    raise RuntimeError(f"Process termination is prohibited: pid={self.pid}")


os.kill = signal_guard
BaseProcess.terminate = prohibit_process_signal
BaseProcess.kill = prohibit_process_signal

SCHEMA_VERSION = 1

# Marker used by bench_ab.py when it has to fall back to parsing stdout.
STDOUT_SENTINEL = "@@Q21_RESULT_JSON@@"

DEFAULT_REPO = "/home/gongji/0z5a-work/qwen21-kernels/repo"
DEFAULT_MODEL = "/home/gongji/0z5a-work/qwen21-kernels/models/Qwen-Image-2.1"
DEFAULT_PROMPT = "A red ceramic teapot on a wooden table."
FETCH_MARKER = ".fetch-complete"

# Keep gongji's GPU7 restriction unless a replacement host has an explicit allowlist.
ALLOWED_DEVICES = tuple(int(index) for index in os.environ.get("Q21_ALLOWED_DEVICES", "7").split(","))

# Exit codes (kept stable: bench_ab.py keys off these).
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_WEIGHTS_MISSING = 3
EXIT_RUN_FAILED = 4

# ---------------------------------------------------------------------------
# Weights preflight
# ---------------------------------------------------------------------------

# Files that must exist for the pipeline to load. Derived from the HF repo
# layout of Qwen/Qwen-Image-2.1 (manifest of the in-flight fetch:
# model_index.json, processor/, scheduler/, text_encoder/, transformer/, vae/).
REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
    "processor/tokenizer_config.json",
    "processor/preprocessor_config.json",
)

# Subfolders whose *.safetensors shards must be present and non-empty.
WEIGHT_SUBFOLDERS = ("transformer", "text_encoder", "vae")


def _shards_for(model_dir: Path, subfolder: str) -> list[str]:
    """Resolve expected shard filenames for one component subfolder."""
    index = model_dir / subfolder / "model.safetensors.index.json"
    if not index.exists():
        index = model_dir / subfolder / "diffusion_pytorch_model.safetensors.index.json"
    names: list[str] = []
    if index.exists():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            names = sorted({Path(v).name for v in weight_map.values()})
        except Exception:
            names = []
    if not names:
        names = sorted(p.name for p in (model_dir / subfolder).glob("*.safetensors"))
    return names


def preflight_weights(model_dir: Path) -> dict:
    """Check that the checkpoint on disk is complete enough to load.

    Returns a dict: {"ok": bool, "missing": [...], "empty": [...],
                     "marker": bool, "shards": {subfolder: [names]}}.
    """
    missing: list[str] = []
    empty: list[str] = []
    shards: dict[str, list[str]] = {}

    if not model_dir.exists():
        return {
            "ok": False,
            "missing": [str(model_dir)],
            "empty": [],
            "marker": False,
            "shards": {},
            "reason": "model directory does not exist",
        }

    for rel in REQUIRED_FILES:
        path = model_dir / rel
        if not path.exists():
            missing.append(rel)
        elif path.stat().st_size == 0:
            empty.append(rel)

    for sub in WEIGHT_SUBFOLDERS:
        names = _shards_for(model_dir, sub)
        shards[sub] = names
        if not names:
            missing.append(f"{sub}/*.safetensors (no shards found)")
            continue
        for name in names:
            path = model_dir / sub / name
            if not path.exists():
                missing.append(f"{sub}/{name}")
            elif path.stat().st_size == 0:
                empty.append(f"{sub}/{name}")

    marker = (model_dir / FETCH_MARKER).exists()
    return {
        "ok": not missing and not empty,
        "missing": missing,
        "empty": empty,
        "marker": marker,
        "shards": shards,
    }


def _missing_weights_error(model_dir: Path, pre: dict, require_marker: bool) -> dict | None:
    """Build the structured error payload when the checkpoint is not ready."""
    if not pre["ok"]:
        n_missing = len(pre["missing"])
        n_empty = len(pre["empty"])
        detail = []
        if pre["missing"]:
            detail.append(f"{n_missing} missing path(s): " + ", ".join(pre["missing"][:8]))
            if n_missing > 8:
                detail.append(f"... and {n_missing - 8} more")
        if pre["empty"]:
            detail.append(f"{n_empty} zero-byte file(s): " + ", ".join(pre["empty"][:8]))
        return {
            "error_code": "WEIGHTS_INCOMPLETE",
            "error_type": "WeightsNotReady",
            "error": (
                f"Qwen-Image-2.1 checkpoint at {model_dir} is incomplete; "
                + "; ".join(detail)
                + ". The download writes into .chunks/ and the marker file "
                f"'{FETCH_MARKER}' is written on completion. "
                "Run harness/check_ready.sh for a full readiness report."
            ),
        }
    if require_marker and not pre["marker"]:
        return {
            "error_code": "WEIGHTS_UNVERIFIED",
            "error_type": "WeightsNotReady",
            "error": (
                f"all expected files exist under {model_dir}, but the completion marker "
                f"'{FETCH_MARKER}' is absent, so the fetch may still be in flight. "
                "Re-run with --allow-unverified-weights to proceed anyway."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# CLI / env
# ---------------------------------------------------------------------------


def _env_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _env_bool(name: str) -> bool | None:
    raw = _env_str(name)
    if raw is None:
        return None
    low = raw.lower()
    if low in ("1", "true", "yes", "on", "y"):
        return True
    if low in ("0", "false", "no", "off", "n"):
        return False
    raise SystemExit(f"invalid boolean for ${name}: {raw!r} (use 1/0)")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_t2i.py",
        description=(
            "Run ONE Qwen-Image-2.1 text-to-image generation through vllm-omni's "
            "offline Omni API and print machine-readable JSON on stdout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Every option can also be supplied through a Q21_* environment variable "
            "(see README.md); explicit CLI flags win over the environment. "
            f"stdout: one JSON object, optionally prefixed with {STDOUT_SENTINEL}."
        ),
    )
    g = p.add_argument_group("workload")
    g.add_argument("--model", default=None, help=f"local checkpoint dir (env Q21_MODEL) [{DEFAULT_MODEL}]")
    g.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt")
    g.add_argument("--negative-prompt", default=None, help="negative prompt (only used when cfg-scale > 1)")
    g.add_argument("--seed", type=int, default=42, help="RNG seed")
    g.add_argument("--image", action="append", default=None, metavar="PATH",
                   help="condition image for edit mode (repeatable; up to 4). Supplying one "
                        "or more switches the request from text-to-image to image edit, which "
                        "makes the prefix cache long.")
    g.add_argument("--height", type=int, default=1024, help="image height (aligned to 32)")
    g.add_argument("--width", type=int, default=1024, help="image width (aligned to 32)")
    g.add_argument("--steps", type=int, default=None, help="num_inference_steps (env Q21_STEPS) [40]")
    g.add_argument("--cfg-scale", type=float, default=1.0, help="true_cfg_scale (1.0 disables CFG)")
    g.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="guidance_scale; leave unset for this model (pipeline picks its own default)",
    )
    g.add_argument("--num-outputs", type=int, default=1, help="num_outputs_per_prompt")
    g.add_argument("--output", default="/home/gongji/0z5a-work/qwen21-kernels/harness/out/t2i.png",
                   help="output PNG path")
    g.add_argument("--json-out", default=None, help="also write the JSON object to this file")
    g.add_argument("--variant", default=None, help="free-form label, echoed into the JSON (env Q21_VARIANT)")

    g2 = p.add_argument_group("engine knobs (env vars in parentheses)")
    g2.add_argument("--repo", default=None, help=f"vllm-omni checkout (env Q21_REPO) [{DEFAULT_REPO}]")
    g2.add_argument("--device", default=None,
                    help="physical GPU index, or a comma list for tensor parallel; sets "
                         "CUDA_VISIBLE_DEVICES (env Q21_DEVICE) [1]")
    g2.add_argument("--attention-backend", default=None,
                    help="diffusion attention backend (env Q21_ATTENTION_BACKEND), e.g. FLASH_ATTN, TORCH_SDPA")
    g2.add_argument("--prefix-kv-cache-dtype", default=None,
                    choices=["none", "auto", "fp8", "fp8_v"],
                    help="cross-step prefix KV storage dtype (env Q21_PREFIX_KV_CACHE_DTYPE)")
    g2.add_argument("--enforce-eager", default=None, action=argparse.BooleanOptionalAction,
                    help="True => eager DiT (no CUDA-graph decode); False/None => CUDA-graph decode "
                         "(env Q21_ENFORCE_EAGER)")
    g2.add_argument("--cpu-offload", default=None, action=argparse.BooleanOptionalAction,
                    help="model-level CPU offload (env Q21_CPU_OFFLOAD)")
    g2.add_argument("--layerwise-offload", default=None, action=argparse.BooleanOptionalAction,
                    help="layerwise (sequential) CPU offload (env Q21_LAYERWISE_OFFLOAD)")
    g2.add_argument("--cache-backend", default=None,
                    help="diffusion cache backend, e.g. cache_dit / tea_cache (env Q21_CACHE_BACKEND)")
    g2.add_argument("--vae-tiling", default=None, action=argparse.BooleanOptionalAction,
                    help="VAE decode tiling (env Q21_VAE_TILING)")
    g2.add_argument("--vae-slicing", default=None, action=argparse.BooleanOptionalAction,
                    help="VAE decode slicing (env Q21_VAE_SLICING)")
    g2.add_argument("--profiler", default=None, action=argparse.BooleanOptionalAction,
                    help="enable the diffusion pipeline profiler => stage_durations "
                         "(env Q21_PROFILER) [on]")
    g2.add_argument("--executor-backend", default=None,
                    help="distributed_executor_backend: uni (in-process worker) or mp (env Q21_EXECUTOR_BACKEND)")
    g2.add_argument("--extra-engine-kwargs", default=None,
                    help="JSON object merged into the Omni(**kwargs) call (env Q21_EXTRA_ENGINE_KWARGS)")
    g2.add_argument("--max-num-seqs", type=int, default=None, help="engine max_num_seqs")

    g3 = p.add_argument_group("harness behaviour")
    g3.add_argument("--repeat", type=int, default=1, help="run the generation N times in THIS process (default 1)")
    g3.add_argument("--cuda-profile", action="store_true", help="mark the second request for Nsight Systems")
    g3.add_argument("--timeout-s", type=float, default=3600.0, help="hard wall-clock budget for the process")
    g3.add_argument("--poll-device-mem", default=True, action=argparse.BooleanOptionalAction,
                    help="poll device-wide used memory with nvidia-smi during the run")
    g3.add_argument("--mem-poll-interval-ms", type=int, default=250, help="device memory poll interval")
    g3.add_argument("--allow-unverified-weights", action="store_true",
                    help="proceed even without the .fetch-complete marker")
    g3.add_argument("--allow-any-device", action="store_true",
                    help="do not refuse reserved GPU indices (0 and 4-7)")
    g3.add_argument("--dry-run", action="store_true",
                    help="SYNTHETIC: skip model load, emit a fake result + a generated placeholder PNG")
    g3.add_argument("--preflight-only", action="store_true",
                    help="check the checkpoint and exit (no torch / vllm import)")
    g3.add_argument("--check-imports", action="store_true",
                    help="report the resolved vllm_omni / torch / vllm import paths and exit")
    g3.add_argument("--no-redirect-stdout", action="store_true",
                    help="do not redirect fd1->fd2 (logs will interleave with the JSON)")
    g3.add_argument("--quiet", action="store_true", help="suppress progress chatter on stderr")
    return p


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge env-var defaults under the explicit CLI values."""
    cfg: dict = {}

    def pick(cli_value, env_name, fallback=None, cast=None):
        if cli_value is not None:
            value = cli_value
        else:
            raw = _env_str(env_name)
            value = fallback if raw is None else raw
        if value is None:
            return None
        if cast is not None and isinstance(value, str):
            value = cast(value)
        return value

    def pick_bool(cli_value, env_name, fallback):
        if cli_value is not None:
            return cli_value
        env = _env_bool(env_name)
        return fallback if env is None else env

    cfg["model"] = pick(args.model, "Q21_MODEL", DEFAULT_MODEL)
    cfg["repo"] = pick(args.repo, "Q21_REPO", DEFAULT_REPO)
    raw_device = str(pick(args.device, "Q21_DEVICE", 1, str)).strip()
    cfg["devices"] = [int(part) for part in raw_device.split(",") if part.strip()]
    if not cfg["devices"]:
        raise SystemExit("--device must name at least one GPU index")
    # `device` stays the first index (the memory poller's anchor); `visible_devices` is what
    # CUDA_VISIBLE_DEVICES gets, so a comma list enables tensor parallel across the allowed GPUs.
    cfg["device"] = cfg["devices"][0]
    cfg["visible_devices"] = ",".join(str(index) for index in cfg["devices"])
    cfg["steps"] = pick(args.steps, "Q21_STEPS", 40, int)
    cfg["attention_backend"] = pick(args.attention_backend, "Q21_ATTENTION_BACKEND", None)
    cfg["prefix_kv_cache_dtype"] = pick(args.prefix_kv_cache_dtype, "Q21_PREFIX_KV_CACHE_DTYPE", "none")
    cfg["enforce_eager"] = pick_bool(args.enforce_eager, "Q21_ENFORCE_EAGER", None)
    cfg["cpu_offload"] = pick_bool(args.cpu_offload, "Q21_CPU_OFFLOAD", False)
    cfg["layerwise_offload"] = pick_bool(args.layerwise_offload, "Q21_LAYERWISE_OFFLOAD", False)
    cfg["cache_backend"] = pick(args.cache_backend, "Q21_CACHE_BACKEND", None)
    cfg["vae_tiling"] = pick_bool(args.vae_tiling, "Q21_VAE_TILING", False)
    cfg["vae_slicing"] = pick_bool(args.vae_slicing, "Q21_VAE_SLICING", False)
    cfg["profiler"] = pick_bool(args.profiler, "Q21_PROFILER", True)
    cfg["executor_backend"] = pick(args.executor_backend, "Q21_EXECUTOR_BACKEND", "uni")
    cfg["variant"] = pick(args.variant, "Q21_VARIANT", "default")
    extra = pick(args.extra_engine_kwargs, "Q21_EXTRA_ENGINE_KWARGS", None)
    if isinstance(extra, str):
        extra = json.loads(extra)
    cfg["extra_engine_kwargs"] = extra or {}
    return cfg


# ---------------------------------------------------------------------------
# Device-wide memory polling (captures the spawned stage subprocess)
# ---------------------------------------------------------------------------


class DeviceMemPoller:
    """Poll device-wide used memory for ONE physical GPU via nvidia-smi.

    The diffusion stage runs in a spawned process, so ``torch.cuda`` counters in
    this process cannot see the model. This poller gives an honest external
    high-water mark. It never touches a GPU outside ``device``.
    """

    def __init__(self, device: int | list[int], interval_ms: int = 250):
        self.devices = device if isinstance(device, list) else [device]
        self.device = self.devices[0]
        self.interval_s = max(0.05, interval_ms / 1000.0)
        self._stop = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.baseline: int | None = None
        self.peak: int | None = None
        self.source = "nvidia-smi"
        self.error: str | None = None
        self._cmd = [
            "nvidia-smi", "-i", ",".join(str(index) for index in self.devices),
            "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        ]

    def _read(self) -> int | None:
        try:
            out = subprocess.run(self._cmd, capture_output=True, text=True)
        except Exception as exc:  # pragma: no cover - depends on host
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        if out.returncode != 0:
            self.error = (out.stderr or "").strip()[:200] or f"exit {out.returncode}"
            return None
        try:
            # One line per polled GPU; report the total used across them.
            lines = [line for line in out.stdout.strip().splitlines() if line.strip()]
            return sum(int(line) for line in lines) * 1024 * 1024
        except Exception as exc:  # pragma: no cover
            self.error = f"unparsable nvidia-smi output {out.stdout!r}: {exc}"
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self.samples.append(value)
                self.peak = value if self.peak is None else max(self.peak, value)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.baseline = self._read()
        if self.baseline is not None:
            self.samples.append(self.baseline)
            self.peak = self.baseline
        self._thread = threading.Thread(target=self._loop, name="q21-mem-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        final = self._read()
        if final is not None:
            self.samples.append(final)
            self.peak = final if self.peak is None else max(self.peak, final)

    def as_dict(self) -> dict:
        delta = None
        if self.peak is not None and self.baseline is not None:
            delta = max(0, self.peak - self.baseline)
        return {
            "enabled": True,
            "source": self.source,
            "gpu_index": self.device,
            "baseline_used_bytes": self.baseline,
            "peak_used_bytes": self.peak,
            "peak_delta_bytes": delta,
            "num_samples": len(self.samples),
            "interval_ms": int(self.interval_s * 1000),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Import-path resolution
# ---------------------------------------------------------------------------


def resolve_imports(repo: str) -> dict:
    """Import vllm_omni/torch/vllm and report the resolved module files."""
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    info: dict = {}
    for name in ("torch", "vllm", "vllm_omni"):
        try:
            mod = __import__(name)
            mod_file = getattr(mod, "__file__", None) or ""
            info[name] = {
                "file": mod_file or None,
                "version": getattr(mod, "__version__", None),
                "from_repo": bool(mod_file) and mod_file.startswith(repo),
            }
        except Exception as exc:
            info[name] = {"file": None, "version": None, "from_repo": False,
                          "error": f"{type(exc).__name__}: {exc}"}
    return info


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_images(outputs) -> list:
    """Pull PIL images out of an OmniRequestOutput list (repo convention)."""
    from diffusers.utils import numpy_to_pil

    import numpy as np

    images = None
    for output in outputs or []:
        images = getattr(output, "images", None)
        if images:
            break

    if not images:
        try:
            from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

            images = extract_images_from_outputs(outputs)
        except Exception:
            images = None

    if not images:
        return []

    normalized = []
    for image in images:
        if isinstance(image, np.ndarray):
            normalized.extend(numpy_to_pil(image))
        else:
            normalized.append(image)
    return normalized


_PHASE_SUFFIXES = (
    (".text_encoder.forward", "text_encoder"),
    (".tokenizer.forward", "tokenizer"),
    (".vae.decode", "vae_decode"),
    (".vae.encode", "vae_encode"),
    (".diffuse", "denoise"),
    (".forward", "pipeline_forward"),
)


def _phase_ms(stage_durations: dict | None) -> dict:
    """Map the profiler's raw keys onto stable phase names (milliseconds)."""
    phases: dict[str, float] = {}
    for key, value in (stage_durations or {}).items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        for suffix, name in _PHASE_SUFFIXES:
            if key.endswith(suffix):
                phases[name] = phases.get(name, 0.0) + seconds * 1000.0
                break
    return {k: round(v, 4) for k, v in phases.items()}


def _torch_peaks() -> tuple[int, int, str]:
    """Best-effort torch.cuda peaks for THIS process."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, 0, "cuda_unavailable"
        return (
            int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()),
            "harness_process",
        )
    except Exception:
        return 0, 0, "unavailable"


def _torch_reset_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Synthetic (dry-run) result
# ---------------------------------------------------------------------------


def _synthetic_result(cfg: dict, args: argparse.Namespace) -> dict:
    """Fabricate a result shape without touching the model.

    Clearly labelled ``synthetic: true``; used to smoke-test the plumbing
    (argument handling, JSON contract, PNG writing, hashing, A/B driver)
    before the weights land.
    """
    from PIL import Image

    rng = random.Random(f"{cfg['variant']}|{args.seed}|{args.height}x{args.width}|{args.steps}")
    base = 900.0 + rng.random() * 400.0
    if cfg["enforce_eager"]:
        base *= 1.15
    if (cfg["prefix_kv_cache_dtype"] or "none") != "none":
        base *= 0.97

    denoise = base * 0.86
    phases = {
        "text_encoder": round(base * 0.055, 4),
        "tokenizer": round(base * 0.002, 4),
        "denoise": round(denoise, 4),
        "vae_decode": round(base * 0.06, 4),
        "pipeline_forward": round(base * 0.995, 4),
    }
    stage_durations = {f"SyntheticPipeline.{k}": v / 1000.0 for k, v in phases.items()}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    img.save(out_path)

    peak_alloc = rng.randrange(1 << 28, 1 << 31)
    peak_reserved = peak_alloc + rng.randrange(1 << 27, 1 << 29)
    return {
        "ok": True,
        "synthetic": True,
        "wall_ms": round(base, 3),
        "phase_ms": phases,
        "stage_durations_s": stage_durations,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "torch_peak_scope": "synthetic",
        "worker_peak_reserved_bytes": peak_reserved,
        "output_image": str(out_path),
        "output_image_sha256": _sha256_file(out_path),
        "image_count": 1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)

    # --- stdout hygiene: fd1 -> fd2, keep the real fd1 for the result -------
    real_stdout_fd = os.dup(1)
    if not args.no_redirect_stdout:
        os.dup2(2, 1)
        sys.stdout = os.fdopen(1, "w", buffering=1)

    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = f"{started_utc}-{uuid.uuid4().hex[:8]}"

    def say(msg: str) -> None:
        if not args.quiet:
            print(f"[run_t2i] {msg}", file=sys.stderr, flush=True)

    poller: DeviceMemPoller | None = None
    model_dir = Path(cfg["model"]).expanduser()

    # Pin the visible device *before* anything can import torch / init CUDA, in
    # every mode (including --check-imports). CUDA_VISIBLE_DEVICES is set for
    # this process and inherited by the spawned diffusion stage process.
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["visible_devices"]

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "synthetic": bool(args.dry_run),
        "run_id": run_id,
        "started_utc": started_utc,
        "variant": cfg["variant"],
        "condition_images": list(args.image) if args.image else [],
        "repo": cfg["repo"],
        "model": str(model_dir),
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "steps": cfg["steps"],
        "cfg_scale": args.cfg_scale,
        "height": args.height,
        "width": args.width,
        "negative_prompt": args.negative_prompt,
        "wall_ms": None,
        "error": None,
        "error_code": None,
        "error_type": None,
    }

    def emit(payload: dict, code: int) -> int:
        # Finalize the device poller *before* serializing so its high-water mark
        # is part of the emitted object.
        if poller is not None:
            poller.stop()
            payload["device_memory"] = poller.as_dict()
        payload.setdefault("wall_ms", None)
        payload["duration_s"] = round(time.time() - started, 3)
        text = json.dumps(payload, sort_keys=True)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n")
        with os.fdopen(os.dup(real_stdout_fd), "w") as fh:
            fh.write(STDOUT_SENTINEL + " " + text + "\n")
            fh.flush()
        os.close(real_stdout_fd)
        return code

    # --- device guard -------------------------------------------------------
    reserved = [index for index in cfg["devices"] if index not in ALLOWED_DEVICES]
    if reserved and not args.allow_any_device:
        result.update({
            "error_code": "DEVICE_RESERVED",
            "error_type": "UsageError",
            "error": (
                f"GPU(s) {reserved} are reserved by the machine owner; this workstream may only "
                f"use {list(ALLOWED_DEVICES)}. Pass --device {ALLOWED_DEVICES[0]} (or --allow-any-device "
                "to override deliberately)."
            ),
        })
        return emit(result, EXIT_USAGE)

    env_subset = {
        k: os.environ.get(k)
        for k in (
            "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "VLLM_ATTENTION_BACKEND",
            "DIFFUSION_ATTENTION_BACKEND", "VLLM_CACHE_ROOT", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "VLLM_LOGGING_LEVEL",
        )
    }
    result["env"] = env_subset
    result["resolved_config"] = {
        k: v for k, v in cfg.items() if k != "extra_engine_kwargs"
    }
    result["extra_engine_kwargs"] = cfg["extra_engine_kwargs"]

    # --- preflight ----------------------------------------------------------
    pre = preflight_weights(model_dir)
    result["weights"] = {
        "dir": str(model_dir),
        "complete": pre["ok"],
        "marker": pre["marker"],
        "missing": pre["missing"],
        "empty": pre["empty"],
        "shards": pre["shards"],
    }

    if args.preflight_only:
        problem = _missing_weights_error(model_dir, pre, require_marker=not args.allow_unverified_weights)
        ready = problem is None
        result["ok"] = ready
        result["ready"] = ready
        if problem:
            result.update(problem)
        result["imports"] = resolve_imports(cfg["repo"]) if args.check_imports else None
        say(f"preflight: complete={pre['ok']} marker={pre['marker']} ready={ready}")
        return emit(result, EXIT_OK if ready else EXIT_WEIGHTS_MISSING)

    if args.check_imports:
        result["imports"] = resolve_imports(cfg["repo"])
        result["ok"] = all(
            not v.get("error") for v in result["imports"].values()
        ) and result["imports"]["vllm_omni"].get("from_repo", False)
        if not result["imports"]["vllm_omni"].get("from_repo"):
            result["error_code"] = "WRONG_VLLM_OMNI"
            result["error_type"] = "ImportError"
            result["error"] = (
                "vllm_omni did not resolve to the repo checkout; set "
                f"PYTHONPATH={cfg['repo']} (imported from "
                f"{result['imports']['vllm_omni'].get('file')!r})."
            )
            return emit(result, EXIT_USAGE)
        say("imports resolved; vllm_omni comes from the repo checkout")
        return emit(result, EXIT_OK)

    problem = _missing_weights_error(model_dir, pre, require_marker=not args.allow_unverified_weights)
    if problem is not None and not args.dry_run:
        # Fail loudly and informatively, never with a raw traceback.
        result.update(problem)
        result["imports"] = None
        say("BLOCKED: checkpoint not ready -> " + str(problem["error"]))
        print(
            "\n*** run_t2i: checkpoint not ready ***\n"
            f"{problem['error']}\n"
            f"*** marker present: {pre['marker']}; missing entries: {len(pre['missing'])} ***\n",
            file=sys.stderr, flush=True,
        )
        return emit(result, EXIT_WEIGHTS_MISSING)

    if args.dry_run:
        say("SYNTHETIC dry run: no model is loaded, timings are fabricated")
        result.update(_synthetic_result(cfg, args))
        result["imports"] = None
        result["engine_kwargs"] = {"__note__": "dry-run: engine never constructed"}
        return emit(result, EXIT_OK)

    # --- real run -----------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["visible_devices"]
    env_subset["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    if cfg["repo"] and cfg["repo"] not in sys.path:
        sys.path.insert(0, cfg["repo"])

    poller: DeviceMemPoller | None = None
    omni = None
    try:
        imports = resolve_imports(cfg["repo"])
        result["imports"] = imports
        if not imports.get("vllm_omni", {}).get("from_repo"):
            result.update({
                "error_code": "WRONG_VLLM_OMNI",
                "error_type": "ImportError",
                "error": (
                    "vllm_omni resolved outside the repo checkout "
                    f"({imports.get('vllm_omni', {}).get('file')!r}); set PYTHONPATH={cfg['repo']}."
                ),
            })
            return emit(result, EXIT_USAGE)
        say(f"vllm_omni <- {imports['vllm_omni']['file']}")

        import torch  # noqa: F401
        from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProcManager

        def wait_for_natural_exit(manager, timeout=None):
            manager.manager_stopped = True
            manager.proc.join()

        StageDiffusionProcManager.shutdown = wait_for_natural_exit
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if os.environ.get("Q21_VLLM_030_COMPAT") == "1":
            from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

            original_priority = CudaOmniPlatform.get_default_ir_op_priority

            def ir_priority(cls, vllm_config):
                priority = original_priority(vllm_config)
                # vLLM 0.30 added this IR op with triton/native providers, not vllm_c.
                priority.gelu_and_mul_sparse = ["triton", "native"]
                return priority

            CudaOmniPlatform.get_default_ir_op_priority = classmethod(ir_priority)
            result["runtime_compatibility"] = "vllm0.30 gelu_and_mul_sparse=triton,native"

        if args.poll_device_mem:
            poller = DeviceMemPoller(cfg["devices"], args.mem_poll_interval_ms)
            poller.start()

        omni_kwargs: dict = {
            "model": str(model_dir),
            "enable_cpu_offload": bool(cfg["cpu_offload"]),
            "enable_layerwise_offload": bool(cfg["layerwise_offload"]),
            "vae_use_tiling": bool(cfg["vae_tiling"]),
            "vae_use_slicing": bool(cfg["vae_slicing"]),
            "enable_diffusion_pipeline_profiler": bool(cfg["profiler"]),
            "log_stats": False,
        }
        if cfg["executor_backend"]:
            omni_kwargs["distributed_executor_backend"] = cfg["executor_backend"]
        if cfg["enforce_eager"] is not None:
            omni_kwargs["enforce_eager"] = bool(cfg["enforce_eager"])
        if cfg["attention_backend"]:
            omni_kwargs["diffusion_attention_backend"] = cfg["attention_backend"]
        if cfg["cache_backend"]:
            omni_kwargs["cache_backend"] = cfg["cache_backend"]
        if args.max_num_seqs is not None:
            omni_kwargs["max_num_seqs"] = args.max_num_seqs
        kv_dtype = (cfg["prefix_kv_cache_dtype"] or "none").lower()
        omni_kwargs["extras"] = {
            "prefix_kv_cache_dtype": None if kv_dtype in ("none", "") else kv_dtype
        }
        omni_kwargs.update(cfg["extra_engine_kwargs"])
        if args.cuda_profile:
            omni_kwargs["profiler_config"] = {"profiler": "cuda"}
        result["engine_kwargs"] = {k: v for k, v in omni_kwargs.items()}

        say("constructing Omni engine (loads the checkpoint into a stage subprocess) ...")
        init_t0 = time.perf_counter()
        omni = Omni(**omni_kwargs)
        init_ms = (time.perf_counter() - init_t0) * 1000.0
        say(f"engine ready in {init_ms:.0f} ms")

        prompt_dict: dict = {"prompt": args.prompt, "modalities": ["image"]}
        if args.image:
            # The Qwen-Image-2.1 pipeline reads condition images from
            # request.prompt["multi_modal_data"]["image"] (pipeline_qwen_image_21.py:192-193).
            from PIL import Image as _PILImage

            images = []
            for path in args.image:
                resolved = Path(path).expanduser()
                if not resolved.is_file():
                    result.update({"error_code": "MISSING_IMAGE", "error_type": "FileNotFoundError",
                                   "error": f"condition image not found: {resolved}"})
                    return emit(result, EXIT_USAGE)
                images.append(_PILImage.open(resolved).convert("RGB"))
            prompt_dict["multi_modal_data"] = {"image": images}
            say(f"edit mode with {len(images)} condition image(s)")
        if args.negative_prompt:
            prompt_dict["negative_prompt"] = args.negative_prompt

        # Determinism: pass ONLY ``seed``. The worker materialises
        # ``torch.Generator(device=...).manual_seed(seed)`` in the stage process
        # (diffusion_model_runner.py:637-645), which avoids shipping a CUDA
        # generator across the process boundary.
        sampling = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            seed=args.seed,
            num_inference_steps=cfg["steps"],
            true_cfg_scale=args.cfg_scale,
            num_outputs_per_prompt=args.num_outputs,
            max_sequence_length=None,
            generator=None,
        )
        if args.guidance_scale is not None:
            sampling.guidance_scale = args.guidance_scale

        runs = []
        for repetition in range(max(1, args.repeat)):
            if args.cuda_profile and repetition == 1:
                omni.start_profile(stages=[0])
            _torch_reset_peaks()
            t0 = time.perf_counter()
            outputs = omni.generate(prompt_dict, sampling_params_list=[sampling])
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if args.cuda_profile and repetition == 1:
                omni.stop_profile(stages=[0])
            alloc, reserved, scope = _torch_peaks()
            entry = {
                "repetition": repetition,
                "wall_ms": round(wall_ms, 3),
                "peak_allocated_bytes": alloc,
                "peak_reserved_bytes": reserved,
                "torch_peak_scope": scope,
                "outputs": outputs,
            }
            runs.append(entry)
            say(f"repetition {repetition}: wall_ms={wall_ms:.1f}")

        last = runs[-1]
        outputs = last["outputs"]
        result["init_ms"] = round(init_ms, 3)
        result["wall_ms"] = last["wall_ms"]
        result["repetitions"] = [
            {k: v for k, v in r.items() if k != "outputs"} for r in runs
        ]
        if len(runs) > 1:
            result["wall_ms_mean"] = round(sum(r["wall_ms"] for r in runs) / len(runs), 3)
            result["wall_ms_median"] = round(
                sorted(r["wall_ms"] for r in runs)[len(runs) // 2], 3
            )
        result["peak_allocated_bytes"] = last["peak_allocated_bytes"]
        result["peak_reserved_bytes"] = last["peak_reserved_bytes"]
        result["torch_peak_scope"] = last["torch_peak_scope"]

        if not outputs:
            raise RuntimeError("omni.generate() returned no outputs")
        first = outputs[0]
        stage_durations = getattr(first, "stage_durations", None) or {}
        result["stage_durations_s"] = {k: round(float(v), 6) for k, v in stage_durations.items()}
        result["phase_ms"] = _phase_ms(stage_durations)
        denoise_ms = result["phase_ms"].get("denoise")
        if denoise_ms and cfg["steps"]:
            result["denoise_ms_per_step"] = round(denoise_ms / cfg["steps"], 4)
        worker_peak_mb = float(getattr(first, "peak_memory_mb", 0.0) or 0.0)
        result["worker_peak_reserved_bytes"] = int(worker_peak_mb * 1024 * 1024)
        metrics = getattr(first, "metrics", None)
        if isinstance(metrics, dict):
            result["worker_metrics"] = {
                k: v for k, v in metrics.items() if not isinstance(v, (bytes, bytearray))
            }

        images = _normalize_images(outputs)
        if not images:
            raise RuntimeError("no images found in the Omni outputs")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(images) == 1:
            images[0].save(out_path)
            saved = [out_path]
        else:
            saved = []
            for idx, img in enumerate(images):
                p = out_path.with_name(f"{out_path.stem}_{idx}{out_path.suffix or '.png'}")
                img.save(p)
                saved.append(p)
        result["output_image"] = str(saved[0])
        result["output_images"] = [str(p) for p in saved]
        result["image_count"] = len(saved)
        result["image_size"] = list(images[0].size)
        result["output_image_sha256"] = _sha256_file(saved[0])
        result["output_image_bytes"] = saved[0].stat().st_size
        result["ok"] = True
        return emit(result, EXIT_OK)

    except BaseException as exc:  # noqa: BLE001 - we always want a JSON verdict
        result["ok"] = False
        result["error_code"] = result.get("error_code") or "RUN_FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc()[-4000:]
        say("run failed: " + result["error"])
        return emit(result, EXIT_RUN_FAILED)
    finally:
        if poller is not None:
            poller.stop()
            result["device_memory"] = poller.as_dict()
        if omni is not None:
            try:
                omni.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
