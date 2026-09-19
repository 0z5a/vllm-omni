# Real-adapter switch E2E preparation

Status: prepared harness, GPU execution pending. This is not a passing K5 result.

`benchmark.py` uses the public sampling API on pinned runtime `698f716`, with the two verified trained adapters documented in `../lora-model-preparation`. Each feature arm must run in a fresh process. The sequence is A → A → B → None → A → A(scale=0) → B(scale=0) → A, repeated for two warmup cycles and five measured cycles at 512×512, 768×512 and 512×512 reentry. All PNGs, timings and hashes are retained. Adapter transitions are part of request latency. The output directory must be new, preventing accidental evidence overwrite.

Prepared entry points cover native, TeaCache, Cache-DiT, Ulysses, Ring, TP and HSDP. A CLI option is not proof that the combination is supported or reaches its intended path. Before scheduling formal quartets, separate worker probes must establish actual injection, rank-local adapter identity and scale, cache reset and real hit/no-hit behavior. The timed harness deliberately does not add intrusive worker probes. Native A/P/P/A paired runs and image comparison are still required; no speed table can yet be populated.

All twelve K5 subcases remain separate. CFG and EP need a compatible real model and trained adapter; this Z-Image fixture does not establish them. Layerwise offload, module offload, VAE parallel and FP8 still need their own actual-path admission and configurations. No case is marked complete by these preparations.

Retain the two task-owned trained adapters until their pending cases finish. Never remove the shared base checkpoint as part of cleanup.
