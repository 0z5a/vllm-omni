# Distributed Layerwise Offload

This document describes distributed layerwise offload (DLO) for diffusion
models. DLO keeps only a small number of DiT blocks on the accelerator and
streams the remaining blocks from host memory. The distributed backend can
either shard those host-side weights across an existing parallel group or keep
the standard loader's rank-local weights and avoid an additional collective.
The rank-local mmap storage design is tracked in
[RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195).

For user-facing commands, see the
[distributed layerwise offloading guide](../../user_guide/diffusion/cpu_offload.md)
and the [Cosmos3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md).

## Status

DLO is implemented for multi-device diffusion execution. The default
AllGather path is the primary path for DP and SP deployments. The
`--dlo-no-use-allgather` path is a rank-local compatibility mode: it is useful
for standard-loader sharding, workstation bring-up, and systems where an
additional DLO collective is undesirable. For explicitly mmap-compatible
models in a TP1, non-HSDP topology, this path shares file-backed DiT checkpoint
pages across processes and keeps only bounded staging buffers private. TP,
HSDP, and other models retain standard-loader rank-local host storage.

The compatibility matrix below describes the current implementation. The
unit-level guards are covered, but not every parallelism combination has a
full model-and-hardware end-to-end test.

## Design

### DLO consumes the existing parallel topology

DLO does not create a new DP, TP, or SP topology. It reads the configured
`DiffusionParallelConfig` and attaches offload hooks to the DiT blocks after the
standard distributed groups have been initialized.

The DLO weight-sharding group is selected as follows:

1. Use the existing DP group when `data_parallel_size > 1`.
2. When DP is one and SP is greater than one, use the SP group.
3. Otherwise, run rank-locally without a DLO process group.

TP is deliberately not used as DLO's AllGather group. HSDP has its own
parameter-sharding lifecycle and is not allowed to be sharded a second time by
DLO's AllGather path.

### AllGather path

With the default `dlo_use_allgather=True`, each rank stores approximately
`1 / group_size` of each streamable block in pinned host memory. The next
block's shard is copied to a device buffer and reconstructed with
`all_gather_into_tensor` on a communication stream while the current block is
executing.

Multi-rank AllGather behavior is unchanged. When the effective DLO group size
is one, the backend now uses rank-local mmap staging; this closes a previous
loader/backend mismatch that could leave parameters on the `meta` device.

```text
Compute:    [Block N]             [Block N+1]          [Block N+2]
H2D:                      [shard N+1]           [shard N+2]
AllGather:                [full N+1]             [full N+2]
Buffers:    [current slot]       [prefetch slot]       [current slot]
```

![DLO double-buffer prefetch pipeline](../figures/dlo/dlo_pipeline.gif)

The backend uses two shared device buffers, so accelerator weight residency is
bounded by the largest streamed blocks rather than the complete model.

When DP is greater than one, the engine can process one request per DP rank in
the same denoising wave. Because AllGather is a collective, all participating
requests must take the same execution path at every denoising step.

### Rank-local path without DLO AllGather

With `--dlo-no-use-allgather`, DLO forces its internal offload shard size to
one and streams complete runtime blocks using H2D copies only. Host storage is
selected independently from that transfer protocol:

- For an explicitly mmap-compatible model with TP1 and no HSDP, each worker
  retains immutable safetensors views backed by the node's shared OS page
  cache. Two pinned host buffers, sized to the largest streamed block, stage
  mmap pages for asynchronous H2D. Checkpoint-to-runtime adapters run while a
  block is packed, so transformed weights do not become a persistent private
  model copy.
- With TP, HSDP, online quantization, or a model without mmap support, the
  regular loader remains responsible for preparing rank-local tensors. This
  preserves TP/HSDP/quantization loader callbacks.

This mode means:

- DP still provides independent replicas, but DLO does not shard weights
  across DP ranks.
- SP still performs its normal activation/attention collectives, but DLO does
  not shard weights across SP ranks.
- TP/HSDP/SP collectives, if configured, are not disabled by this flag; only
  DLO's additional weight AllGather is disabled.
- Pure DP deployments with a compatible mmap checkpoint share the streamed
  DiT's file-backed pages. Other components and fallback paths may still keep
  rank-private host weights.
- The scheduler does not require a synchronized DP request wave for DLO.

The mmap source remains immutable in both transfer modes. AllGather copies a
persistent `1 / group_size` shard from it and then closes the mappings;
rank-local mode keeps the mappings open and copies a complete block through
bounded staging without adding a collective.

The current implementation packs mmap views into a staging slot synchronously
on the host; the subsequent pinned H2D transfer overlaps with block compute.
This keeps host memory bounded but makes CPU packing bandwidth part of the
rank-local path's performance profile.

## Parallelism compatibility

| Parallelism | DLO + AllGather | DLO without AllGather |
|---|---|---|
| **DP** | Supported primary path. DLO shards host weights across the DP group and can run DP multi-concurrency. | Supported rank-local path. DP replicas remain independent; no DLO weight collective. Compatible TP1 models share file-backed DiT pages across replicas. |
| **SP** | Supported in the implementation. With DP=1, DLO uses the SP group for host-weight sharding; SP still shards sequence/activation work. | SP remains active, but DLO keeps standard-loader rank-local weights and adds no SP weight collective. |
| **TP > 1** | Unsupported in the DLO mmap path. TP-aware loader callbacks are bypassed, so the backend rejects this configuration when it enters that path. | Standard loading is retained and TP-local tensors can be streamed. This is the intended compatibility path, but it still needs broader model and hardware validation. |
| **HSDP** | Rejected. HSDP has already sharded parameters, so DLO AllGather would double-shard them. | Accepted by configuration. HSDP owns parameter sharding and its own gathers; DLO only stages rank-local parameters. End-to-end coverage is limited. |

### Combined dimensions

- **DP + SP:** DLO uses the DP group for weight sharding when DP is greater
  than one; SP continues to use its own sequence-parallel group. If DP is one,
  the SP group becomes DLO's sharding group in AllGather mode.
- **DP + TP/SP without AllGather:** standard model loading defines the
  rank-local tensor layout. DLO adds no cross-DP, cross-TP, or cross-SP weight
  collective.
- **HSDP + SP:** the general parallel configuration permits HSDP over SP, but
  DLO must use `--dlo-no-use-allgather`. HSDP remains responsible for weight
  materialization and synchronization.
- **HSDP + DP or TP:** rejected independently by the diffusion parallel
  configuration.

## Request and loading constraints

AllGather DP multi-concurrency requires:

- explicit `num_inference_steps`;
- the same `num_inference_steps` for all requests in a wave; and
- identical request arguments that affect the collective execution path.

The no-AllGather path does not impose these DLO-specific synchronized-wave
requirements.

The mmap loader is used when the model explicitly declares a compatible
checkpoint layout. It is independent of AllGather for TP1/non-HSDP execution.
Models can declare a per-parameter checkpoint-to-runtime adapter; the adapter
must preserve dtype and element count so it can be applied while packing one
staging block. MiniMax-H3 uses this mechanism for grouped QKV. Online
quantization and TP/HSDP no-AllGather execution use the regular loader path.

### Adding mmap support to another model

A new pipeline should opt in only after its ordinary loader contract can be
represented without materializing a private full-model copy:

1. Populate `weights_sources` with safetensors component sources and prefixes
   that place every DiT checkpoint key in the pipeline parameter namespace.
2. If prefixed checkpoint keys already equal runtime parameter names, declare
   `_supports_mmap_loading = True`. If names differ, provide
   `_remap_ckpt_key` to map checkpoint names to runtime names (or return
   `None` for intentionally ignored keys).
3. If a parameter needs a TP1 layout conversion, attach a callable
   `mmap_weight_transform` to that parameter during model construction. The
   transform must preserve dtype and element count because it runs while one
   bounded block is packed.
4. Verify that mmap and the regular loader produce the same runtime tensors,
   that every expected DiT tensor maps successfully, and that TP, HSDP, and
   online-quantized configurations retain their standard-loader behavior.

Setting `_supports_mmap_loading = True` alone is not sufficient: explicit
opt-ins fail during initialization if any expected DiT tensor remains on the
`meta` device after checkpoint mapping.

## Validation coverage

Current source-level validation includes:

- HSDP + DLO + AllGather rejection;
- HSDP + DLO without AllGather acceptance at configuration level;
- TP rejection in the DLO+AllGather mmap path;
- resident-layer requests requiring no-AllGather;
- rank-local mmap source retention, layout transforms, and bounded host
  staging;
- DP request-wave validation for denoising-step compatibility;
- sharding, double-buffer, AllGather-size, and heterogeneous-block regression
  tests.

The highest-value missing coverage is end-to-end numerical comparison against
ordinary layerwise offload for DP+SP, TP+no-AllGather, and HSDP+SP+no-AllGather
on the target CUDA/NCCL or CANN/HCCL hardware.

## Recommendations

- Use **DP + DLO AllGather** for the supported throughput and host-memory
  scaling path.
- Use **SP + DLO AllGather** for long-sequence workloads when DP concurrency is
  not the goal.
- Use **no-AllGather** when replicas must avoid DLO collectives. TP1 models with
  mmap support can share DiT checkpoint pages; TP/HSDP and unsupported models
  retain the higher-memory standard-loader path.
- Prefer **HSDP alone** for production HSDP deployments until the combined
  HSDP + DLO no-AllGather path has broader end-to-end coverage.
