# K6/K7 preparation on pinned source 698f716

## Existing CPU contracts

Command (CUDA hidden):

```sh
python -m pytest -q -m cpu \
  tests/diffusion/test_diffusion_step_pipeline.py \
  tests/diffusion/distributed/test_hsdp.py \
  tests/diffusion/distributed/test_expert_parallel_layout.py
```

Result: **60 passed, 3 deselected**, 14 deprecation warnings, 17.77 seconds (`cpu.log`). The three hardware step tests were excluded. These suites cover CPU/mocked runner and scheduler lifecycles, interrupts/abort rescheduling and cleanup, HSDP configuration/sharding contracts, and mocked expert group mappings. They do not establish full-model step/offload/HSDP correctness, GPU collectives, actual expert routing, or speed.

## Model eligibility for K6

The pinned ZImage pipeline has no step execution declaration. It is not used to infer step support from LoRA/cache/parallel tests. Helios declares `supports_step_execution = True` and implements `prepare_encode`, `denoise_step`, `step_scheduler` and `post_decode`; its transformer declares `_hsdp_shard_conditions`. Existing Helios weights are already present, so this candidate needs no duplicate download.

These source declarations select a candidate; they do not replace actual loading or lifecycle tests. Full K6-H/K6-M validation must still test initialization, several steps, pause/resume, cancellation, repeated requests and cleanup, with actual per-rank progress and parameter/offload observations. The existing Helios VAE/step comparisons in the GPU queue do not by themselves fulfill all of those requirements.

## K7-H boundary

Result: **5 passed**, 14 deprecation warnings, 12.29 seconds.

`test_ep_hsdp_guard.py` exercises the real `DiffusionParallelConfig` rejection of EP+HSDP, both alone and with Ulysses/Ring/CFG overlays, and keeps standalone HSDP valid. It does not remove the guard or claim EP+HSDP runtime support. A failed SSH connection before the test started is retained separately; the retry result is in `ep-hsdp-guard-retry.log`.

K7-U/R/C/T still require actual MoE routing/collective checks and full-model admission. Configuration tests and small layers cannot substitute for their E2E results. No speed number is available from these CPU checks.
