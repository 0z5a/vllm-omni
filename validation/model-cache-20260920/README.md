# Prepared checkpoints

All checks here read safetensors headers, file sizes and index-to-shard tensor-key mappings. Repository revisions identify the selected snapshots; these are not GPU-fit or full-pipeline validation claims.

| Checkpoint | Revision | Preparation |
| --- | --- | --- |
| FLUX.2-klein-4B | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | 19 Diffusers/tokenizer/license files; 4 safetensors, 15,964,212,614 bytes; all required selected files and indexed keys verified. Standalone duplicate checkpoint omitted |
| SenseNova-U1-8B-MoT | `bfa9b436503cb8aed4f2bc60e3236710cc77468d` | 8 shards, 35,104,831,192 bytes, 1,116 indexed tensors verified |
| Existing LTX-2 cache | `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` | Reused existing cache; 35 safetensors, 143,646,795,306 bytes, consistent download metadata revision and no partial files |
| Existing Helios cache | `b991c0379a018f4de3227d95468237f56066f5bb` | Reused existing cache; 12 safetensors, 80,481,086,028 bytes, consistent download metadata revision and no partial files |

LTX's cached text encoder contains both `model-*` and `diffusion_pytorch_model-*` shard sets. The total cached bytes must not be interpreted as a single loaded parameter footprint. The shared LTX/Helios paths are owned by other work and are retained. Newly downloaded weights remain needed for pending E2E runs; completed Ovis cleanup is documented separately.
