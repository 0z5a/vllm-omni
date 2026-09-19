# SenseNova RoPE batch broadcasting

Source: `fb3d6d54be21740398f1c8f824361b3850d9ec67`, base `698f7160125d3071b8c0eef69b1b03fa8dfba766`.
One L20 (GPU 3), FP32/BF16, sequence lengths 1/7/32, batch 2, shared or independent RoPE tables.

| Validation | Baseline | Corrected |
| --- | --- | --- |
| Batched reference regression | 6 failed, 6 passed | 12 passed |
| Full kernel test file | Not rerun | 44 passed |
| Changed-file pre-commit | Not rerun | All passed |

This is a correctness fix; test execution time is not an inference speedup measurement. GPU 2 had unrelated activity during the single-GPU correctness checks.

The HSDP probe with `7db1da7` plus this exact kernel correction (equivalent source changes to rebased `e94dafe`) passes all eight rank records: batch 1/2/3/1, three autoregressive steps, denoising lengths 4/6/4, exact outputs and unchanged prefix KV. This uses small real model modules, not the full checkpoint. Original failing HSDP logs remain preserved.
