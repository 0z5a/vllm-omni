# LoRA and offload CPU prerequisite regression

Pinned source: `698f7160125d3071b8c0eef69b1b03fa8dfba766`, immutable remote `/dev/shm/0z5a-compat-698f716`. CUDA devices were hidden; `OMP_NUM_THREADS=2`.

| Suite selection | Passed | Skipped | Runtime | Evidence level |
|---|---:|---:|---:|---|
| LoRA manager, base linear, module residency, layerwise backend, sequential backend | 119 | 4 | 1.42 s | Existing CPU contracts with synthetic layers and mocked device operations |

The raw pytest output is retained in `cpu.log`. These tests cover adapter scaling/rank changes, suspend/resume, partial packed masks, failed-binding cleanup, supported-module discovery and offload lifecycle contracts. They do not prove combined full-model LoRA/offload execution or speed. Four accelerator-dependent pin-memory tests were skipped while CUDA was hidden. K5.9/K5.10 and all other full E2E gates remain pending.

Reproduction on the stated source/runtime:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=/dev/shm/0z5a-compat-698f716 \
/home/kxqandccx/vllm-omni-7753-hidream-20260918/venv/bin/python -m pytest -q \
 tests/diffusion/lora/test_lora_manager.py \
 tests/diffusion/lora/test_base_linear.py \
 tests/diffusion/offloader/test_module_residency.py \
 tests/diffusion/offloader/test_layerwise_backend.py \
 tests/diffusion/offloader/test_sequential_backend.py
```

## CFG adapter prefetch

The public [Edit-R1 adapter](https://huggingface.co/chestnutlzj/Edit-R1-Qwen-Image-Edit-2509/tree/6f58a70502ea6560da338c5af91d7a3d156077ee) was prefetched for the already available Qwen-Image-Edit-2509 base. Its model card names that exact base. Revision `6f58a70502ea6560da338c5af91d7a3d156077ee` contains 960 tensors, rank 32, alpha 64; the 377,615,760-byte safetensors file matches upstream LFS SHA-256 `ffccd3511ba8a198c2d5f7a630e5af5a7ad5465c77411a15e528bfd1aa289060`. Download logs, immutable upstream manifest and all tensor shapes are retained.

Task-owned remote path: `/home/kxqandccx/omni-1217-20260919/models/lora/edit-r1`. Keep it while K5.5 is pending, then remove the owned adapter after evidence is complete. Loading/target matching, a second suitable trained adapter, actual CFG rank behavior and full E2E remain pending. Downloading this adapter does not establish support. Another searched adapter, PaCo, documents the original Qwen-Image-Edit base and was not substituted for a 2509-trained adapter.
