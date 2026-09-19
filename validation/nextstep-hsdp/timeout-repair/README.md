# NextStep HSDP startup budget repair

The first HSDP2 startup on source `43c4e79` successfully loaded the complete checkpoint and entered its 1,024-token dummy generation. The observed rate settled near 1.98 seconds/token, implying roughly 34 minutes for startup generation alone. The harness had only a 900-second initialization deadline and a 14,400-second full-arm deadline, insufficient for this observed path and the full 21-request matrix.

This startup was intentionally terminated before a timeout; it was not a crashed model, a completed request, or a measured E2E comparison. Its raw log and GPU telemetry are preserved. All owned workers and waiting controllers were verified terminal and GPUs 2/3 returned to 14 MiB before restarting. Other tasks were untouched.

The replacement driver uses 7,200 seconds for both initialization limits and 86,400 seconds per arm. It retains the complete model, resolutions, two warmups/five timed repetitions, seed, dtype and topology. The new controller starts H0/H1/TP1; the completed TP0 and separate profile are preserved. It then runs SenseNova, Layered, Qwen Edit, TeaCache, ERNIE VAE, Helios, GLM, DBCache and the separate TeaCache no-hit probes sequentially. No speed result is inferred from the progress bar.

The earlier four waiting controllers are terminal. The replacement controller is PID 681180; its log is `/home/kxqandccx/omni-1217-20260919/resume-long-hsdp.log`. Full HSDP completion, correctness and memory/latency tradeoff remain pending.
