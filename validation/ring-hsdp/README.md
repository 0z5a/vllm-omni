# Ring attention with HSDP

This K3 harness compares two-rank Ring attention against the same topology with HSDP on the pinned ZImage checkpoint. Untimed probes require the Ring strategy on every attention layer, exact positional tensor and output parity, and real DTensor/FSDP residency. Four fresh timing arms run in native/HSDP/HSDP/native order with two warmups and five measured requests per case.

GPU execution waits for the earlier validated queues and uses physical GPUs 2 and 3 only.
