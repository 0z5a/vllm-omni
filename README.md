# Encoder consolidation validation

- [Qwen Image 2.1: dual-order A100 complete-request comparison](qwen/REPORT.md). 13.25% lower latency with six BF16 head layers resident and an additional 2.44 GiB reserved memory. This is not an equal-residency fusion-only gain. PR #7945 remains Draft; external ReadTheDocs inventory HTTP429 is not a passing CI result.
- [FLUX Kontext and Klein: separate hardware and execution cohorts](flux/REPORT.md). Includes negative results, actual graph controls and image/hidden-state differences.

Original request records, derived reports, representative contact sheets and canonical archive indexes are included. All indexed original files were rehashed before copying. Full native traces, hidden-state tensors and the remaining raw images are retained in the original SHA-backed archives; they are not all uploaded here. Canonical indexes refer to those complete archives, while `published-sha256.json` lists the compact files actually published. No model weights are included. Ovis's unfinished matrix is not represented as passed.

## Reproduction inputs

Each cohort includes its exact original benchmark, audit and shutdown helpers under `original/`, alongside its request/config records. These are preserved evidence of the executed harness, not production changes. Restore the tested source revision from the report, match the recorded package versions and model revision, and adapt recorded task paths to your local checkpoint/workspace. Use each cohort's own hardware, precision, offload and compilation settings; the different cohorts are not interchangeable.
