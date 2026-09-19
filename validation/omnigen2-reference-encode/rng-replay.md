# Controlled posterior RNG replay

The pipeline calls `posterior.sample()` without the request generator. The initial raw-reference baseline produced different PNG hashes for identical prompts, inputs and request seed 142. That interrupted run is retained remotely under `evidence-uncontrolled-posterior-rng` and is excluded from paired correctness claims.

The replay harness uses the public `diffusion_model_runner_cls` override. `ReferenceRngRunner` sets the global Torch RNG to the request seed on each worker before execution. This controls the existing posterior sampler without editing either pipeline implementation or changing reference order. Both arms use this same harness and include its small setup cost in E2E latency. It is controlled replay, not a claim that the unmodified public seed API controls the posterior sampler.

The candidate is still `f2c1684` and its baseline `15840a6`; both use VAE degree 2, Ulysses 2 advanced_uaa, identical BF16 weights and 30 steps. The first two replay outputs were identical; the full quartet is still running and has no final correctness or speed claim yet.
