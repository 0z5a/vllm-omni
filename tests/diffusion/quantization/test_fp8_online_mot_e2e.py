# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E on the real MoT production path.

The ``Omni`` entrypoint cannot run in this environment -- the installed
vllm_omni 0.28.0 imports ``vllm.inputs.preprocess``, which vLLM 0.29.0 removed --
so this test drives the same code the engine drives, one level down:

    MoTRowParallelLinear._mot_gemm_dispatch
      -> _mot_gemm_fp8_w8a8
           -> fp8_online.scaled_fp8_quant      <-- the change under test
           -> invoke_mot_gemm                  <-- the real Triton MoT kernel

Nothing is stubbed: real FP8 weights, real per-tensor scales, the real Triton
kernel, and the real per-token quantizer.

It is a pytest test rather than a standalone script because vLLM's online
quantizers call ``get_current_vllm_config()`` while creating weights, and the
session-scoped ``default_vllm_config`` fixture is what establishes that context
in the supported order.  The three arms are run inside one process but with the
fast path explicitly toggled through the module's kill switch, and the timed
region excludes all weight construction, Triton JIT, and warmup.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import (  # noqa: E402
    DeviceConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config  # noqa: E402

# The real engine exports these for its workers; the MoT layer's weight
# parameters read the TP rank, so a single-rank rendezvous has to be declared.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")

INPUT_SIZE = int(os.environ.get("MOT_INPUT", "3072"))
OUTPUT_SIZE = int(os.environ.get("MOT_OUTPUT", "3072"))
TOKENS = int(os.environ.get("MOT_TOKENS", "512"))
VAE_TOKENS = int(os.environ.get("MOT_VAE_TOKENS", "256"))
ITERS = int(os.environ.get("MOT_ITERS", "50"))
REPS = int(os.environ.get("MOT_REPS", "7"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="needs a CUDA device with sm_80+",
)


def _cached_model_path() -> str:
    """A locally cached snapshot with a config.json, for ModelConfig.

    ModelConfig validates its model argument on construction even though this
    test only reads ``model_config.dtype`` from it.
    """
    import glob

    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for snap in sorted(glob.glob(os.path.join(hub, "models--*", "snapshots", "*"))):
        if os.path.isfile(os.path.join(snap, "config.json")):
            return snap
    raise RuntimeError("no cached HF snapshot with config.json found")


@pytest.fixture(scope="module", autouse=True)
def e2e_distributed():
    """Single-rank process groups.

    vLLM weight parameters read the tensor-parallel rank while being created, so
    the groups must exist.  Mirrors tests/diffusion/distributed/test_comm.py.
    """
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    init_distributed_environment(backend="nccl")
    ensure_model_parallel_initialized(1, 1)
    yield


@pytest.fixture(scope="module")
def e2e_vllm_config():
    """Config context for MoT layer construction.

    ``default_vllm_config`` (autouse, session scope) establishes the context;
    this object is what the layer's online quantizer reads ``dtype`` from.
    """
    return VllmConfig(
        device_config=DeviceConfig(device="cuda"),
        model_config=ModelConfig(
            model=os.environ.get("MOT_CONFIG_MODEL") or _cached_model_path(),
            dtype=torch.bfloat16,
            seed=0,
        ),
    )


def _fp8_weight(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(*shape, device="cuda", dtype=torch.float32, generator=g) * 0.05
    return w.to(torch.float8_e4m3fn)


def _spin_up(fn, seconds=8.0):
    """Warm the GPU to a stable clock before any timed measurement.

    This part idles at 180 MHz, so the first measured arm otherwise runs at a
    much lower clock than the later ones -- which is exactly what made an
    earlier attempt show a fake 1.43x with an A/A ratio of 0.59.
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(100):
            fn()
        torch.accelerator.synchronize()


def _median_us(fn, iters, reps):
    samples = []
    for _ in range(reps):
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(samples), samples


def test_mot_module_fast_path(default_vllm_config, e2e_vllm_config):
    from vllm_omni.diffusion.models.bagel.mot.mot_row_parallel_linear import (
        MoTRowParallelLinear,
    )
    from vllm_omni.quantization import fp8_online

    with set_current_vllm_config(e2e_vllm_config):
        from vllm.distributed.parallel_state import ensure_model_parallel_initialized

        ensure_model_parallel_initialized(1, 1)
        layer = MoTRowParallelLinear(
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            bias=False,
            vae_bias=False,
            params_dtype=torch.bfloat16,
            # MoT's FP8 W8A8 path is reached from pre-quantized (serialized)
            # FP8 weights, which is also what makes create_weights produce
            # concrete tensors instead of meta-device placeholders.
            quant_config=Fp8Config(is_checkpoint_fp8_serialized=True),
            prefix="mot.e2e",
            reduce_results=False,
            disable_tp=True,
        ).cuda()
    layer.eval()

    with torch.no_grad():
        layer.weight.data.copy_(_fp8_weight((OUTPUT_SIZE, INPUT_SIZE), 1))
        layer.gen_exp.weight.data.copy_(_fp8_weight((OUTPUT_SIZE, INPUT_SIZE), 2))
        for holder in (layer, layer.gen_exp):
            ws = getattr(holder, "weight_scale", None)
            if ws is not None:
                ws.data.fill_(0.05)

    g = torch.Generator(device="cuda").manual_seed(7)
    # Activations must be bf16 to match what the transformer emits; the widened
    # dynamic range (a few large tokens among small ones) is what makes the
    # per-token scale non-degenerate.
    x = (
        torch.randn(
            TOKENS + VAE_TOKENS, INPUT_SIZE, device="cuda",
            dtype=torch.bfloat16, generator=g,
        )
        * torch.exp(
            torch.randn(
                TOKENS + VAE_TOKENS, 1, device="cuda",
                dtype=torch.bfloat16, generator=g,
            ) * 1.5
        )
    ).to(torch.bfloat16)
    text_idx = torch.arange(0, TOKENS, device="cuda", dtype=torch.long)
    vae_idx = torch.arange(TOKENS, TOKENS + VAE_TOKENS, device="cuda", dtype=torch.long)

    def run_once():
        with set_current_vllm_config(e2e_vllm_config), torch.no_grad():
            return layer._mot_gemm_fp8_w8a8(x, text_idx, vae_idx, None, None)

    print("\n[diag] weight dtype      =", layer.weight.dtype)
    print("[diag] gen_exp dtype      =", layer.gen_exp.weight.dtype)
    print("[diag] weight_scale       =", getattr(layer, "weight_scale", None))
    print("[diag] gen_exp weight_scale =", getattr(layer.gen_exp, "weight_scale", None))
    print("[diag] input dtype        =", x.dtype)

    # Warmup: Triton JIT/autotune and the extension build must not be timed.
    out_ref = run_once()
    torch.accelerator.synchronize()

    n_rows = TOKENS + VAE_TOKENS
    in_envelope = n_rows <= fp8_online.FASTPATH_MAX_ROWS

    # Gate check: supported() is the dispatch gate, so it must be True exactly
    # inside the measured envelope.
    fp8_online._DISABLED = False
    assert fp8_online.supported(x) is in_envelope

    # Balanced interleaved sweep: A1 P P A1, so a monotonic clock/power drift
    # cancels between the two arms instead of being attributed to the patch.
    def set_arm(disabled):
        os.environ["VLLM_OMNI_FP8_ONLINE_DISABLE"] = "1" if disabled else "0"
        fp8_online._DISABLED = disabled

    _spin_up(run_once, seconds=8.0)

    times = {"A1": [], "P": []}
    outs = {}
    for slot, disabled in (("A1", True), ("P", False), ("P", False), ("A1", True)):
        set_arm(disabled)
        t, _ = _median_us(run_once, ITERS, REPS)
        times[slot].append(t)
        outs[slot] = run_once().clone()
        torch.accelerator.synchronize()

    set_arm(False)
    t_a1 = statistics.mean(times["A1"])
    t_p = statistics.mean(times["P"])
    t_p0, _s_p0 = _median_us(run_once, ITERS, REPS)   # module on, gate forced off
    set_arm(True)
    t_p0, _s_p0 = _median_us(run_once, ITERS, REPS)
    out_p0 = run_once().clone()
    torch.accelerator.synchronize()

    out_a1 = outs["A1"]
    out_p = outs["P"]
    s_a1 = times["A1"]
    s_p = times["P"]
    aa_ratio = times["A1"][1] / times["A1"][0]

    # The MoT output must not change.  Triton accumulation order can differ
    # between launches, so compare against the same-arm repeat and allow a
    # tight relative tolerance rather than requiring bit equality.
    def rel_close(a, b, tol=2e-3):
        return bool(torch.allclose(a.float(), b.float(), rtol=tol, atol=tol))

    print(f"\nMoT path  rows={n_rows} in_envelope={in_envelope}")
    print(f"  A1={t_a1:.2f}us  P={t_p:.2f}us  P0={t_p0:.2f}us")
    print(f"  A1/P = {t_a1 / t_p:.3f}x")
    print(f"  P0/A1 = {t_p0 / t_a1:.3f}x  (must be ~1.0)")
    print(f"  samples A1={['%.2f' % v for v in s_a1]}")
    print(f"  samples P ={['%.2f' % v for v in s_p]}")

    assert rel_close(out_a1, out_p0), "enabling the module must not change output"
    assert rel_close(out_a1, out_p), "fast path must not change the MoT output"
    assert out_p.shape == (TOKENS + VAE_TOKENS, OUTPUT_SIZE)

    # The quantizer itself is the part under test and must be byte-exact.
    from vllm import _custom_ops as ops

    x2 = x.view(-1, x.shape[-1])
    ref_o, ref_s = ops.scaled_fp8_quant(x2, None, use_per_token_if_dynamic=True)
    got_o, got_s = fp8_online.scaled_fp8_quant(x2, None, use_per_token_if_dynamic=True)
    assert torch.equal(ref_o.view(torch.uint8), got_o.view(torch.uint8))
    assert torch.equal(ref_s, got_s)

    # A/A sanity after spin-up: the two A1 slots must agree, otherwise the
    # A1/P ratio is not interpretable and the run must be discarded.
    print(f"  ROWS={n_rows} A1/P={t_a1 / t_p:.4f} AA={aa_ratio:.4f} "
          f"A1={t_a1:.3f} P={t_p:.3f}")
    print(f"  A1 slots={['%.2f' % v for v in times['A1']]} "
          f"P slots={['%.2f' % v for v in times['P']]}")
    assert 0.9 < aa_ratio < 1.1, (
        f"host drifted during the measurement (A/A={aa_ratio:.3f}); "
        "discard this run"
    )
