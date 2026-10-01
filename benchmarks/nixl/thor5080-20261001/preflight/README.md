# RTX 5080 / Jetson Thor continuation preflight

The remaining full-checkpoint Hunyuan validation moves to an RTX 5080 AR
producer and Jetson Thor DiT/VAE consumer. These checks use task-private
environments with readonly Torch installations; they do not constitute a
completed model E2E or a model speedup measurement.

| Check | Execution | Result | Model speedup |
| --- | --- | --- | --- |
| Hunyuan request, routing and native NIXL ownership regressions | 5080 host CPU, 76 cases | 76 passed, zero skips, 281.883 s including setup | Not measured |
| Same Hunyuan/native NIXL regressions | Thor CPU, 76 cases | 76 passed, zero skips, 15.447 s including setup | Not measured |
| Disk-backed native weight offload | RTX 5080, six BF16 linear forwards | Exact outputs; modified weights restored to the same backing file | Not measured |
| Native UCX READ across architectures | 5080 host DRAM → Thor DRAM, 1 MiB | Exact bytes; both processes exited normally without a CUDA context | Not measured |
| Native UCX VRAM READ across architectures | RTX 5080 → Thor, 1 MiB | Exact bytes; Workers and controller exited normally; both GPU leases returned | Not measured |
| Remote DiT normal shutdown delivery | Both hosts' CPUs, peer connects after 1 s | Shutdown received; contexts closed; tests exited normally without a CUDA context | Not measured |
| Full AR → 50-step DiT → VAE | RTX 5080 + Thor | Pending checkpoint completion and GPU availability | Pending |

The offload check also verifies FP32, BF16 and FP8 CPU values and strides.
GPU pinning and UVA are disabled for the disk-backed deployment. The test
does not load the full FP8 MoE checkpoint.

The DRAM and VRAM proofs use native NIXL 1.3.0 with UCX TCP through task-private
SSH relays. Both compare the complete 1 MiB payload against its reference.
Transfer payload SHA-256 is
`fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83`.

The paired VRAM check obtains both real host locks and checks fresh GPU UUIDs,
boot IDs, empty compute lists and the existing Thor queue before allocating
CUDA tensors. Its lock scopes last 51.384 seconds on the 5080 and 57.671 seconds
on Thor, including paired admission, initialization, handshake and transfer.
These are not isolated transfer latencies or model speedup measurements.
Successful proofs, source hashes and post-exit ownership checks are in
[`vram-peer-v2`](../vram-peer-v2/).

The first VRAM attempt fails: its 10-second completion ACK deadline lets the
producer exit while the receiver still reports `PROC`. Restoring a 45-second
ACK deadline and fixing buffered stdout handling in the paired controller
permits the successful run. Its recorded interval from receiver registration
to adding the remote agent is about 32.7 seconds. The failed run and its natural
exits remain preserved in [`vram-peer-v1`](../vram-peer-v1/).

The 5080 has all 18 checkpoint shards, verified against pinned LFS hashes and
totalling 86,177,283,726 bytes. The downloader exits normally. Thor's remaining
checkpoint download awaits a storage allocation that respects the existing
queue; the full 50-step replay is still pending.

The shutdown check exercises the real `StageDiffusionClient` sockets and
encoder with a task-private flush helper. It verifies delayed-peer delivery
on one host; cross-host model shutdown remains pending. Its CPU entry disables
discovery of the unused optional EP binary, which cannot load against the
readonly Torch build. The deployment uses TP1 without expert parallelism.

Thor's private overlay matches all 108 selected dependency pins, including
Transformers 5.14.1, tokenizers 0.22.2 and Diffusers 0.40.0. Both hosts use
vLLM 0.30.0 and NIXL 1.3.0. Readonly Torch is 2.13.0+cu130 on Thor and
2.13.0+cu132 on the 5080; this is not the complete earlier H20 environment.
The CPU suite durations include imports and different host filesystems and
caches, so their ratio is excluded from performance comparisons. The initial
Thor collection failed because the private source copy omitted test helpers;
copying the existing helpers resolved collection before this passing run.

The production source is unchanged from `18f49695af1d542a0c4becea44da5a6af02afd47`.
Raw JUnit, transport logs, offload helpers and SHA-256 manifests are included
here. The earlier H20 synchronization diagnostic remains diagnostic evidence;
it is excluded from accepted model performance results.
