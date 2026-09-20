# K6 full Helios lifecycle preparation

Status: prepared and queued, not GPU validated. Controller PID `1999496` waits for the existing LoRA parallel controller `1432273` to finish successfully. Only physical GPUs 2 and 3 are used. All 33 parent scopes remain open to their own acceptance gates.

Runtime is the immutable `/dev/shm/0z5a-step-lifecycle-bbd391d`: pinned `698f716` plus terminal cleanup `463eb88` and failure consensus `17d3837`. The combined source passed 186 CPU tests, with 11 hardware cases deselected. These are not full-model K6 results.

Full checkpoint: existing shared Helios revision `b991c0379a018f4de3227d95468237f56066f5bb`, already downloaded and previously verified. The checkpoint index declares `HeliosPyramidPipeline`; the native registry resolves it to `HeliosPipeline`. No model index is rewritten. Shared weights are retained while these tests remain pending.

| Configuration | GPUs | Step | HSDP | Module offload | GPU status |
|---|---:|---|---|---|---|
| native | 1 | No | No | No | Pending |
| step | 1 | Yes | No | No | Pending |
| hsdp | 2 | No | Yes | No | Pending |
| step-hsdp | 2 | Yes | Yes | No | Pending |
| module | 1 | No | No | DiT + text encoder | Pending |
| step-module | 1 | Yes | No | DiT + text encoder | Pending |

All use the complete pipeline: text encoder, DiT, VAE and video postprocessing. Fixed input is a red block moving across a blue table; seed 142, BF16 DiT/encoder, FP32 VAE, eager mode, VAE tiling, 640×384, guidance 1, pyramid steps [10,10,10]. Frame sequence is 33 → 66 → 33 for chunk/re-entry coverage. Step mode enables streaming and concatenates all returned chunks rather than comparing only the final chunk.

Each configuration first runs an independent probe process. Step probes additionally execute pause/resume, cancellation, repeat after cancellation, one-rank preparation failure, and repeat after failure. Pause inserts two empty executor waves after the third actual step: retained request tensors, history, RNG, parameter placement and transfer records must remain unchanged. Cancellation uses the engine abort path. Failure injection occurs before pipeline preparation on the last rank; it does not claim recovery from failure inside a collective. All ranks record scheduled IDs and output step/finished states, and their wave sequences must agree.

Probes observe actual first-block/encoder/VAE execution with CUDA-resident parameters, FSDP module count and DTensor residency, and successful module transfer events. The full real worker source hash and probe hash are checked. They require exact full-video equality to the native path and exact repeat/re-entry output. A failed gate stops this matrix for investigation; no reduced frames, smaller checkpoint, disabled feature or relaxed quality threshold is substituted.

After probe gates, twelve fresh timing arms run in forward/reverse configuration order. Each arm has 2 warmups + 5 measured requests for each of the three cases (252 timing/warmup requests total). Timing includes engine admission, all generation stages, transport, chunk assembly and video postprocessing; artifact hashing/compression and intrusive probe RPCs are excluded. Raw videos are preserved as content-addressed NPZ files. GPU UUID/memory/utilization/power are sampled separately.

`verify.py` checks every saved video hash, shape, finiteness, expected request count, repeat identity, feature activation and lifecycle invariant before generating speed/latency/GPU-second Markdown tables. No speed values exist yet. Single-GPU admission and full-model equivalence still require actual execution.

Local Ruff and shell syntax checks pass. Actual remote runtime imports, six full-checkpoint config enrichments, worker extension method names and empty scheduler wave construction pass with GPUs hidden. All five harness files and seven runtime source files are hash checked before the queue starts. The initial import assertion incorrectly expected the native class name instead of the checkpoint's registered alias; `config-import.log` retains that failure and `config-import-final.log` records its correction.
