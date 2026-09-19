# NextStep rectangular E2E failure and correction

Real TP2 generation at source `9545f40` completed seven 512² requests with identical output hashes, then generated 1,536 AR tokens for 768×512 and failed in square-only unpatchify. The original evidence is retained remotely under `nextstep-validation/evidence-unpatchify-failure`; it is an incomplete E2E run.

Independent correction `8e43ef6`, [PR 7853](https://github.com/vllm-project/vllm-omni/pull/7853), passes explicit patch-grid height/width to unpatchify. Its regression exercises actual pipeline forward and real patchify/unpatchify with square, portrait and landscape shapes, batch 1/2 and nontrivial scaling/shift. Baseline: 4 failures, 2 passes. Fixed complete NextStep CPU directory: 16 passes. All changed-file pre-commit checks pass.

HSDP work is rebased to `43c4e79` on this fix, with both comparison arms using the same source. The rebased source archive SHA-256 is `adcf600e293b62347d1edbe10e8ffebde80068ff7f9bf9306a4a25454b39e107`, verified after transfer; the rebased CPU directory has 16 passes. Fresh E2E and the separate stage profile remain pending. Existing [PR 7847](https://github.com/vllm-project/vllm-omni/pull/7847) is updated with this dependency.
