# Rebased Ovis full-pipeline E2E

Author: 0z5a. Candidate `97d2a75`, upstream `698f7160125d3071b8c0eef69b1b03fa8dfba766`. Checkpoint revision `41be1c5821a92c970d63d7eb595a2fd3fe32b22e`.

All six fresh-process arms completed on GPU 2/3. All 84 PNG files were checked locally against their raw SHA256 records; every output within each resolution is identical across native, blockwise HSDP and request residency. CPU lifecycle tests and real two-rank lifecycle/collective-count checks passed before this run; their logs are included.

[Speed comparison](evidence/speed-comparison.md) reports both HSDP and native comparisons. It is a shared-host observation: N0, P0 and P1 contain additional active compute PIDs outside the worker set. Raw process telemetry is retained, including the other PIDs. There is one balanced block and no cross-session confidence interval. This run establishes successful rebased E2E and output parity, not an isolated speed guarantee.

To regenerate the table, run `python audit_processes.py evidence` followed by `python summarize_rebased.py evidence`. Both scripts use the saved records; the latter also hashes every PNG.
