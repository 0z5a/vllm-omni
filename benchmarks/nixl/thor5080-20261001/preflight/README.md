# RTX 5080 / Jetson Thor continuation preflight

The remaining full-checkpoint Hunyuan validation moves to an RTX 5080 AR
producer and Jetson Thor DiT/VAE consumer. These checks use task-private
environments with readonly Torch installations; they do not constitute a
completed model E2E or a model speedup measurement.

| Check | Execution | Result | Model speedup |
| --- | --- | --- | --- |
| Hunyuan request, routing and native NIXL ownership regressions | 5080 host CPU, 76 cases | 76 passed, zero skips, 281.883 s including setup | Not measured |
| Disk-backed native weight offload | RTX 5080, six BF16 linear forwards | Exact outputs; modified weights restored to the same backing file | Not measured |
| Native UCX READ across architectures | 5080 host DRAM → Thor DRAM, 1 MiB | Exact bytes; both processes exited normally without a CUDA context | Not measured |
| Remote DiT normal shutdown delivery | 5080 host CPU, peer connects after 1 s | Shutdown received; contexts closed; test exited normally without a CUDA context | Not measured |
| Full AR → 50-step DiT → VAE | RTX 5080 + Thor | Pending checkpoint completion and GPU availability | Pending |

The offload check also verifies FP32, BF16 and FP8 CPU values and strides.
GPU pinning and UVA are disabled for the disk-backed deployment. The test
does not load the full FP8 MoE checkpoint.

The DRAM proof uses native NIXL 1.3.0 with UCX TCP through task-private SSH
relays. It does not establish that the same route supports native VRAM READ;
that check remains pending. Transfer payload SHA-256 is
`fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83`.

The shutdown check exercises the real `StageDiffusionClient` sockets and
encoder with a task-private flush helper. It verifies delayed-peer delivery
on one host; cross-host model shutdown remains pending. Its CPU entry disables
discovery of the unused optional EP binary, which cannot load against the
readonly Torch build. The deployment uses TP1 without expert parallelism.

The production source is unchanged from `18f49695af1d542a0c4becea44da5a6af02afd47`.
Raw JUnit, transport logs, offload helpers and SHA-256 manifests are included
here. The earlier H20 synchronization diagnostic remains diagnostic evidence;
it is excluded from accepted model performance results.
