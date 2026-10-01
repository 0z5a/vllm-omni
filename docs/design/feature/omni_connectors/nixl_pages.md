# NIXL handoff into native KV pages

The optional page capability extends the existing Omni `NixlConnector`.
`OmniNixlKVConnector` borrows native Scheduler/Worker hooks and delegates every
registration and READ to that same connector's agent. It accepts TP1 → TP1,
one uniform full-attention group, matching model revision, dtype, head geometry,
block size and resolved physical layout. Configuration selects this path
explicitly; ordinary `put/get` payloads keep their existing interface.

The source Scheduler returns delayed free from `request_finished`, retaining
its original native block references. The Worker waits for CUDA events recorded
after each layer's last cache write before publishing descriptors. An idle AR
engine keeps stepping while those exports remain pending, so publication and
completion do not depend on another generation request arriving.

The diffusion Scheduler atomically reserves all CFG rows through
`DiffusionKVCacheManager`. It passes the existing allocation generation and
real block IDs to the Worker. Geometry is checked at the producer before
claims are installed. Both CFG claims enter the source lease before
the first descriptor reply. One batched NIXL READ writes each row directly
into those reserved blocks, including the final physical page. The model only
marks the reusable logical prefix computed and overwrites its suffix normally.
The model adapter's `prompt_token_ids` bounds that logical prefix; the complete
static prefix can include image headers that differ from the AR sequence.
The source ticket carries its computed token IDs, and the receiver only marks
the common token prefix computed. Equal prefix lengths alone do not establish
that the AR and DiT templates agree.
This also applies to a CFG row with zero reusable tokens: it reserves the
physical source pages, receives them, and recomputes the entire logical row.

Pools register their underlying allocations once. Native four-dimensional
views use their actual data pointer, storage offset and strides; separate K/V
views use two regions per physical block. Dense page geometry is checked before
submission. Transfers do not pack source tensors, allocate full receive tensors
or change the native allocator. A claimed offer can be submitted only once.

Only a terminal `DONE` permits transfer-handle release and the exact
generation/claim ACK. A lost ACK keeps ownership until retry succeeds. Unknown
or failed submission states retain handles, registrations and both page
allocations. Timeouts do not cancel DMA. The destination remains reserved after
READ completion until native computation finishes; late completion after an
abort follows that same retirement rule. A Worker abort before submission
cancels its claimed offers through an idempotent ACK, preserving other CFG
readers. An active or ambiguous READ cannot be cancelled this way.
Connector close retains active leases
and leaves its control plane running until ownership drains.

The validation tools compare equal physical bytes across ordinary NIXL
allocation/scatter, local GPU copy and direct pages, then separately compare
matching full-checkpoint Hunyuan AR → paged DiT → VAE requests. The ordinary
model reference lives only in `tools/nixl_page_reference.py` and uses the same
native reservations and attention kernels. It shares claim decoding, destination
reservation checks and the active-read registry with the page path; only the
ordinary allocation/scatter operation is different. Full-model acceptance requires
completed page bytes, registration reuse, no ordinary `get` fallback, drained
leases, matching ordinary-path pixels and the existing Hunyuan local recompute
pixel accuracy thresholds. Two GPUs on one host establish no cross-node RDMA
or heterogeneous TP/SP result.

LLM stages accept native vLLM `offload_config`, including nested `uva` or
`prefetch` settings; the structured config projects their fields into native
EngineArgs. Hunyuan constructs its FP32-routing MoE blocks inside `make_layers`,
so the native offloader sees the final parameters. It no longer replaces those
blocks after offloading or briefly allocates a second set of expert weights.
AR and DiT use the same FP32 router, top-k selection and model-dtype routing
weights; transferring a prefix requires those model computations to agree.
The matched three-path CPU-offload deployment is recorded in
[`offload-deploy.yaml`](../../../../benchmarks/nixl/h20x2-20261001/offload-deploy.yaml).

The current integration does not provide cancellation before Worker metadata
dispatch, reconnect recovery after losing the Worker control channel, or
validated model concurrency above one. All three full-model paths complete,
but their pixel accuracy gates remain unresolved in the
[execution results](nixl_h20_hunyuan_results.md).

Related upstream work: [generic NIXL](https://github.com/vllm-project/vllm-omni/pull/6093),
[structured payloads](https://github.com/vllm-project/vllm-omni/pull/6264),
[roadmap item](https://github.com/vllm-project/vllm-omni/issues/8264).
