# A2/A4/A5/H1 E2E queue

Status: prepared behind the existing GPU 2/3 queue. Completion requires all four authenticated checkpoints, sixteen fresh engine processes, 336 generated images, and `evidence/RESULTS.md`.

FLUX.1 schnell and dev are reported separately. Kontext uses a fixed generated conditioning image and reports its edit path independently. SD3.5 Medium compares BF16 TP2 with BF16 HSDP2; FP8 is not enabled. Each comparison uses two warmups and five measured requests for 512×512, 1024×768, and a 512×512 reentry, with two feature processes bracketed by two reference processes.

The generated Markdown table contains median latency, speedup, latency reduction, and RGB output delta. Raw JSONL, images, process logs, GPU telemetry, source SHAs, and exact checkpoint revisions remain alongside it.
