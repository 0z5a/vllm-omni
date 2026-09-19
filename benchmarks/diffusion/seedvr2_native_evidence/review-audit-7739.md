# Review audit: #7739 at 9a62e04ea8e4bca41d69ad75c07322db13a35f82

Reviewer/owner: 0z5a. Audit date: 2026-09-20. Raw GitHub reviews and inline comments are retained alongside this file. This audit does not approve or merge the PR.

| Finding | Source / regression evidence | Disposition |
| --- | --- | --- |
| Packed-varlen backend capability | `nadit.py:326` resolves backend capability before dispatch; unsupported capability uses grouped SDPA. Existing backend dispatch tests are included in the native CPU regression. Native full-checkpoint runs explicitly request grouped SDPA. | Addressed in parent. FP16 FA3 kernel support is separate and not claimed. |
| Duplicate/unreferenced public helpers | Production `na_ops.py:91,113` calls `joint_cu_seqlens`; `na_ops.py:302` calls `global_window_mean`. Old unused helpers removed. Weighted reduction and empty-rank tests exercise shared implementation. | Addressed in parent. |
| Strict checkpoint loading and RoPE key conversion | Worker reference loader rejects missing/unexpected keys after the one explicit suffix normalization. Native loader rejects missing, duplicate, unexpected, and shape-mismatched keys. Full 32-layer real checkpoint loads in actual engine and HTTP service. | Key claim addressed. Historical reference snapshot revision remains unresolved pending fresh pinned-source validation. |
| Window-SP identity on other models | Parent only documents the caveat. Native integration adds engine identity rejection before `_init_process_hooks`; 7 regression cases prove unsupported rejection and default pass-through. | Implemented locally; not yet published as integration PR. |

Personally checked: source call sites, existing full CPU regression, strict released checkpoint loading, full-weight engine SP1/2/4 parity, native HTTP video/audio/PTS, malformed request recovery. Small SP parity cases do not establish large-video scaling, memory capacity or worker-fault recovery. Native serving source is still uncommitted and its recipe is limited to C0 whole-clip semantics.

Publication dependency: coordinate VAE/offline/serving scope with MinhaoLi0318, who proposed the issue's remaining P0 work. Do not present integration ownership as exclusive.
