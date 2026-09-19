# Reference identity recovered by content verification

On 2026-09-20 the actual remote reference directories used for VAE/NaDiT validation were compared to immutable Git trees fetched through the GitHub API:

| Reference | Immutable commit | Exact tracked-file matches | Changed / missing |
| --- | --- | ---: | ---: |
| numz/ComfyUI-SeedVR2_VideoUpscaler | 4490bd1f482e026674543386bb2a4d176da245b9 | 126 | 0 / 0 |
| ByteDance-Seed/SeedVR | e4de8c24441a67e1b7df56abea10645059bb1185 | 85 | 0 / 0 |

The comparison computes Git blob SHA1 using the standard `blob <length>\0<bytes>` representation and checks every tracked file against the pinned recursive tree. Each file also has a SHA256 in `*-full-snapshot-comparison.json`. The Python subset additionally matches95/95 and73/73 respectively.

This establishes content identity of the current executed reference snapshots. It does not change the historical fact that they were originally fetched from main tarballs without a recorded commit. No reference file was changed to manufacture a match. Historical parity evidence retains its original snapshot hash; new reports can cite these verified immutable commits.
