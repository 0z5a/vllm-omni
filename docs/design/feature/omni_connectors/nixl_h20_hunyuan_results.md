# Hunyuan full-model validation on H20

Checkpoint: `feizhai123/hunyuan-image3-modelopt-fp8`, revision
`d9a3991e01fcaefe0fb14169e8c9d04fc126c4e1`.
The downloaded checkpoint contains all 18 weight shards, totalling
86,177,283,726 bytes. Both stages use the same FP8 weights, BF16 native KV,
TP1, eager execution, Triton MoE and paged FlashAttention.
Requests use 512 × 512 output, 50 diffusion steps, guidance 2 and seeds
1234/1235 for two fixed prompts. Each prompt has one warmup and three
measured requests.

The current replay aligns AR and DiT with the checkpoint's `pretrain`
template and uses the ordinary reference without diagnostic snapshots.
Full-model acceptance remains pending:

| Current path | Completed / measured | Median seconds | Speedup vs ordinary | Status |
| --- | ---: | ---: | ---: | --- |
| Local prefix recomputation | 8 / 6 | 14.654 | Pending | Complete |
| Ordinary NIXL | 8 / 6 | 14.645 | 1.000× observed | Pixel comparison failed; Worker exit unconfirmed |
| Native-page READ | 0 / 0 | Pending | Pending | Awaiting prefix diagnosis and GPU window |

The saved local PNG hashes and shapes were rechecked offline, and each
prompt's four images have identical pixels. Recovered ordinary records contain eight images and zero transport errors,
with eight puts, sixteen gets, one registration per Worker and drained
ownership. The image gate fails: dog warmup PSNR/SSIM is 35.99/0.98366,
dog repeats 12.47/0.05883 and coffee 10.63/0.06073. Ordinary shutdown
retained the DiT Worker after an Orchestrator timeout; its old exit is
unconfirmed. These observed timings do not establish model speedup.
Current records are in
[`template-aligned`](../../../../benchmarks/nixl/h20x2-20261001/template-aligned/).

On the replacement machine, a fixed-version private environment matches all
278 locked package versions and passes pip check and 76 CPU cases. Full-model
prefix tracing completed four images each for local and ordinary, with both
paths' Workers confirmed to exit naturally. Ordinary still fails: dog warmup
PSNR/SSIM 39.206/0.99161, dog repeat 12.463/0.05892, coffee 10.621/0.06055.
All 32 valid source KV layers repeat exactly, and received positive prefixes
and negative BOS match their source bytes. Captured first-step writes and
first denoise projections repeat exactly.
These synchronized diagnostic timings do not establish a speedup. Images,
logs, byte comparisons, source hashes and environment audit are preserved in
[`replacement-machine diagnosis`](../../../../benchmarks/nixl/h20x2-20261001/recovery-prefix-diagnostic/).

A subsequent ordinary-only diagnosis saved full tensors at all 50 steps,
introducing CPU copies and CUDA synchronization. All four images completed;
both Workers exited naturally and both GPUs were released. For each prompt,
all 50 predictions, scheduler inputs/outputs/sigmas, selected layers' Q/K/V
hashes and VAE input/output repeat exactly. The saved pipeline tensors also
reproduce the actual PNG pixels exactly through the installed postprocessor.

| Replacement-machine diagnosis | Completed / measured | Median seconds | Speedup vs ordinary | Image quality vs matched local |
| --- | ---: | ---: | --- | --- |
| Local, prefix trace | 4 / 2 | 14.907 | Not isolated | Reference |
| Ordinary, prefix trace | 4 / 2 | 14.743 | Not isolated | Failed |
| Ordinary, full step trace | 4 / 2 | 19.404 | Not isolated | Dog 43.329 dB / 0.99576; coffee 47.659 dB / 0.99706 |

The full-step synchronization changes the observed failure and satisfies the
image thresholds for these four diagnostic images. It does not establish a
production fix or accepted model speedup. Raw 817-file traces are backed up;
compact images, comparison results, timings and SHA256 manifest are in
[`step synchronization diagnosis`](../../../../benchmarks/nixl/h20x2-20261001/step-synchronization-diagnostic/).
Matched local/ordinary/page-READ runs without diagnostic synchronization remain
required. Further validation moves to RTX 5080 and Jetson Thor; their timings
will be reported separately from H20.

The matched deployment uses native UVA offload with a 96 GiB AR budget,
768 MiB AR KV, 2 GiB DiT KV and an AR batch limit of 512 tokens. This allows
both new stages to run while the original AR Worker retains its claimed lease.
The original Worker is not signalled or terminated. Both new Workers exit
through their normal shutdown protocol after each path.

**Historical diagnostic: accuracy failed for the earlier source and input
template; these times are not an accepted model speedup.** All three paths generated eight images,
including two warmups. Timing ratios compare the same CPU-offload deployment.
The ordinary diagnostic additionally records source KV snapshots, so these
timing ratios do not isolate transport performance.

| Full-model path | Completed / measured requests | Median seconds | Observed ratio vs ordinary | Acceptance |
| --- | ---: | ---: | ---: | --- |
| Local prefix recomputation | 8 / 6 | 38.958 | 0.995× | Reference |
| Ordinary NIXL into native pages | 8 / 6 | 38.773 | 1.000× | Pixel comparison failed |
| NIXL READ into native pages | 8 / 6 | 38.749 | 1.001× | Pixel comparison failed |

The first dog image reaches PSNR 36.036 dB and SSIM 0.98635 against local
recomputation. Coffee and repeated requests fail the PSNR ≥30 dB / SSIM ≥0.97
gate, and ordinary/pages pixels are not identical. Repeated source KV snapshots
are identical over every valid token in all 32 layers for all six measured
requests. Token semantics and destination consumption were subsequently
corrected and checked by the diagnostic below; matched replay remains pending.
The weights remain until those gates pass.

A subsequent CPU regression exposed router downcasting under an outer BF16
autocast: both AR and DiT selected the wrong expert for close FP32 scores.
The gate matmul now explicitly disables autocast. All five routing tests pass,
including the two cases that failed before this change. Because the DiT
computation changed, the local baseline and both transfer paths require replay;
the table above remains evidence for the earlier source manifests.

The token-ID/autocast diagnostic subsequently completed four 50-step images.
Each prompt's repeated image is pixel-identical to its warmup. Against the
earlier local images, dog/coffee PSNR is 42.55/36.18 dB and SSIM is
0.99498/0.98332. All eight CFG receives are byte-identical to their source;
all sixteen captured first-layer rows have exact KV writes and slot mappings,
with CPU-reference attention RMSE at most 0.00001023. This diagnostic does
not replace the matched three-path gate.

Its token trace also exposes an input-template mismatch: the AR validation
builder used Instruct role labels while this checkpoint configures `pretrain`.
Validation now builds AR inputs with the same checkpoint-configured tokenizer
template as DiT. The AR preflight and E2E driver share that builder. Real
tokenizer prefix checks pass for both prompts under both supported templates.
The next matched local/ordinary/pages replay uses these corrected inputs and
the normal reference connector without CPU snapshots. Diagnostic images,
byte-proof summaries, CPU reference results and source hashes are preserved in
[`token-autocast-diagnostic`](../../../../benchmarks/nixl/h20x2-20261001/token-autocast-diagnostic/).

The checkpoint-template local replay completed eight 50-step images, with a
14.654-second median across six measured requests. Each prompt's four images
are pixel-identical. The AR stage emits four tokens per request with the
checkpoint's pretrain template, so these timings are a new input baseline;
the change from the earlier Instruct-input times is not a transfer speedup.
Both Workers exited naturally. Matched ordinary/pages runs remain pending.

For token-checked NIXL text-to-image requests, the adapter now exposes the
entire stable text prefix before the generated-image timestep. The bridge
still stops reuse at the first token mismatch and at the source's computed
length. This permits raw prompts without a CoT terminator; conditioned-image
and legacy connector boundaries retain their previous behavior. The related
layout, routing and native-bridge/reference suites pass 76 CPU cases.

All 24 images, timings, metrics, accuracy scores and per-path source hashes are
preserved in
[`routing-diagnostic`](../../../../benchmarks/nixl/h20x2-20261001/routing-diagnostic/).

Native transport counters report eight exports, sixteen page READs, one pool
registration per Worker, no ordinary get fallback, no transport errors and
zero active reads, leases or destination reservations at completion. These
ownership checks do not establish pixel correctness.

Hunyuan now constructs its final MoE modules before the native offloader wraps
layers, and both AR and DiT share FP32 routing with model-dtype top-k weights.
The full-checkpoint offload AR load/forward preflight passed. Related routing,
FP8 and physical-prefix regressions passed 355 CPU cases. The latest token-ID
boundary coverage passed 336 CPU cases with three GPU cases deselected; config
and engine projection suites passed 280 and 98 cases respectively, each with
one existing Qwen skip. Focused typing for the changed bridge and shared router
passed. The broader config type check retains 27 diagnostics on unchanged lines;
no environment or dependency changes were made to suppress them.

The earlier resident local run completed eight requests with an 18.587-second
median. It used different memory placement and predates the routing correction,
so it is retained as historical evidence and excluded from these timing ratios.

The consolidated code was also measured with the same 8 MiB
payload, 20 warmups per repeat and 1,002 measured samples per path:

| Transfer path | p50 ms | p95 ms | p99 ms | Speedup vs ordinary | Receive tensor allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary NIXL | 5.435 | 25.464 | 25.511 | 1.00× | 8 MiB |
| Local GPU copy | 0.507 | 0.757 | 0.804 | 10.72× | 0 |
| NIXL page READ | 1.858 | 2.142 | 2.226 | 2.93× | 0 |

Each pool registered once across 1,062 transfers including warmups, with
zero transport errors and all ownership drained before natural exit.
This rerun shared the host with the retained AR Worker, which held VRAM
and polled its pending lease. It measures transfer behavior, not model speed.
Raw samples and counters use the `page-consolidation-*` filenames, with
`consolidation-source-hashes.json` recording the tested source. The previous
geometry/cancellation run (`page-integration-*`) measured 2.51×; shared-host
latency varies, so these runs do not isolate a speed change from refactoring.

The remaining full-model replay moves to an RTX 5080 AR producer and Jetson
Thor DiT/VAE consumer. Both hosts pass the same 76 CPU regressions and normal
shutdown delivery checks; the 5080 also passes six exact
BF16 disk-backed native offload forwards, and cross-architecture 1 MiB
native UCX DRAM and VRAM READs with exact bytes and normal process exits.
The paired VRAM check returns both host leases after 51.384 and 57.671 seconds,
including admission and handshake. All 18 pinned model shards are available
on the 5080. Thor checkpoint completion and the full replay remain pending;
these preflight checks
add no model speedup claim. Evidence and the status table are preserved in
[`5080/Thor preflight`](../../../../benchmarks/nixl/thor5080-20261001/preflight/).

The completed transfer benchmark and raw timings are recorded separately in
[page transfer results](nixl_h20_page_results.md). The local request timings,
checkpoint manifest, example images and JUnit evidence are committed under
[`benchmarks/nixl/h20x2-20261001`](../../../../benchmarks/nixl/h20x2-20261001/).
