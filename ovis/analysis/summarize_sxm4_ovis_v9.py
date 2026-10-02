"""Require the full dual-GPU matrix and lifecycle audits before reporting."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    index = json.loads((root / "sha256-index.json").read_text())
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in index.items()
    )
    gate = json.loads(
        (root / "evidence/ovis-public-full-v9-quality-gate.json").read_text()
    )
    assert (
        root / "evidence/ovis-public-full-v9-quality-gate.exit"
    ).read_text().strip() == "0"
    assert gate["encoder_backend"] == "MarlinFP8ScaledMMLinearKernel"
    assert gate["native_kernel_verified"] and gate["images_reviewed"]
    assert (
        gate["public_kernel_control"]
        == "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel"
    )
    assert not gate["kernel_class_patched"]
    queue = root / "evidence/a100-ovis-full-v9"
    assert (queue / "exit").read_text().strip() == "0"
    assert (queue / "completed").is_file()
    plan = json.loads((root / "inputs/ovis-full-queue-v9.json").read_text())
    assert plan["devices"] == "0,1" and len(plan["steps"]) == 24
    manifest_path = root / "ovis/models/Ovis-Image-7B/.verified-complete"
    model = json.loads(manifest_path.read_text())
    assert model["repo_id"] == "ATH-MaaS/Ovis-Image-7B"
    assert model["revision"] == "41be1c5821a92c970d63d7eb595a2fd3fe32b22e"
    assert len(model["files"]) == 19
    source_manifest = json.loads(
        (root / "ovis/evidence/source-manifest.json").read_text()
    )
    assert source_manifest["commit"] == plan["validation"]["source"]
    rows = []
    for index, step in enumerate(plan["steps"]):
        assert (queue / f"step-{index}.exit").read_text().strip() == "0"
        argv = step["argv"]
        output = Path(argv[argv.index("--output") + 1]).name
        directory = root / "ovis/evidence" / output
        assert (directory / "completed").is_file()
        config = json.loads((directory / "config.json").read_text())
        settings = config["arguments"]
        assert (
            config["steps"] == 30
            and config["seed"] == 142
            and config["guidance_scale"] == 4.0
        )
        assert len(config["prompts"]) == 3
        assert config["omni_config"]["dtype"] == "bfloat16"
        assert config["omni_config"]["diffusion_attention_config"] == {
            "default": {"backend": "TORCH_SDPA"}
        }
        assert settings["variant"] in ("bf16", "encoder")
        assert settings["cfg_size"] == 2 and settings["offload"] == "none"
        assert settings["eager"] and settings["weight_load_threads"] == 1
        records = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        assert len(records) == 8 and [record["index"] for record in records] == list(
            range(8)
        )
        assert [record["warmup"] for record in records] == [True, True] + [False] * 6
        assert Counter(record["prompt_index"] for record in records[2:]) == {
            0: 2,
            1: 2,
            2: 2,
        }
        for record in records:
            digest = hashlib.sha256(
                (directory / record["image"]).read_bytes()
            ).hexdigest()
            assert digest == record["image_sha256"]
        workers = json.loads((directory / "worker-after.json").read_text())[0]
        assert len(workers) == 2 and {worker["rank"] for worker in workers} == {0, 1}
        for worker in workers:
            assert (
                worker["sources"][
                    "vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image"
                ]["sha256"]
                == source_manifest["files"][
                    "vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py"
                ]
            )
            assert worker == json.loads(
                (
                    directory / f"worker-after-ranks-rank{worker['rank']}.json"
                ).read_text()
            )
            projections = [
                p for p in worker["projections"] if p["name"].startswith("text_encoder")
            ]
            dit_projections = [
                p for p in worker["projections"] if p["name"].startswith("transformer.")
            ]
            assert len(dit_projections) == 45 and all(
                p["dtype"] == "torch.bfloat16"
                and p["method"] == "UnquantizedLinearMethod"
                for p in dit_projections
            )
            assert set(worker["vae_dtypes"]) == {"torch.bfloat16"}
            if settings["variant"] == "encoder":
                assert len(projections) == 196
                assert all(
                    p["backend"] == "MarlinFP8ScaledMMLinearKernel"
                    and p["dtype"] == "torch.int32"
                    for p in projections
                )
            else:
                # The HF BF16 encoder uses native nn.Linear, not LinearBase.
                assert not projections
                assert len(worker["encoder_native_linears"]) == 196
                assert all(
                    p["dtype"] == "torch.bfloat16"
                    for p in worker["encoder_native_linears"]
                )
                assert set(worker["encoder_parameter_dtypes"]) == {"torch.bfloat16"}
        assert all(
            worker["torch_compile"]["enforce_eager"]
            and not worker["torch_compile"]["compiled_forwards"]
            for worker in workers
        )
        audits = json.loads((directory / "hsdp-after.json").read_text())[0]
        assert len(audits) == 2 and {audit["rank"] for audit in audits} == {0, 1}
        mode = settings["hsdp"]
        for audit in audits:
            assert audit == json.loads(
                (directory / f"hsdp-after-ranks-rank{audit['rank']}.json").read_text()
            )
            assert set(audit["encoder_activation_dtypes"]) == {"torch.bfloat16"}
            assert audit["encoder_activation_dtypes"]["torch.bfloat16"] > 0
            assert len(audit["encoder_calls"]) == 2
            assert all(call["cuda_ms"] > 0 for call in audit["encoder_calls"])
            assert len(audit["before_vae"]) == 1
            if mode == "none":
                assert not audit["gathers"] and not audit["after"]
                continue
            counts = [
                count
                for name, count in audit["gathers"].items()
                if "transformer_blocks." in name
            ]
            assert len(counts) == 33 and all(
                count == (1 if mode == "resident" else 30) for count in counts
            )
            assert len(audit["after"]) == 34 and all(
                row["sharded"] and row["reshard_after_forward"]
                for row in audit["after"]
            )
            assert len(audit["before_vae"][0]) == 34 and all(
                row["sharded"] for row in audit["before_vae"][0]
            )
        if mode == "resident":
            exceptions = json.loads((directory / "hsdp-exception.json").read_text())[0]
            assert len(exceptions) == 2
            for rank in exceptions:
                assert rank == json.loads(
                    (
                        directory / f"hsdp-exception-ranks-rank{rank['rank']}.json"
                    ).read_text()
                )
                assert len(rank["after_injected_exception"]) == 34 and all(
                    row["sharded"] and row["reshard_after_forward"]
                    for row in rank["after_injected_exception"]
                )
        native = []
        for rank in range(2):
            trace = directory / f"cuda-trace-rank{rank}.json"
            events = json.loads(trace.read_text())["traceEvents"]
            kernels = Counter(
                event["name"] for event in events if event.get("cat") == "kernel"
            )
            assert kernels, "A trace file without GPU kernels does not prove execution"
            if settings["variant"] == "encoder":
                assert any("marlin" in name.lower() for name in kernels)
            native.append(
                {
                    "rank": rank,
                    "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    "kernel_counts": dict(kernels),
                    "cuda_graph_launches": sum(
                        event.get("name", "").startswith("cudaGraphLaunch")
                        for event in events
                    ),
                }
            )
        timings = [record["wall_ms"] for record in records[2:]]
        assert all(value > 0 for value in timings)
        rows.append(
            {
                "output": output,
                "resolution": f"{settings['height']}x{settings['width']}",
                "variant": settings["variant"],
                "hsdp": mode,
                "timed_ms": timings,
                "native_profiles": native,
                "runtime_by_rank": [
                    {
                        "rank": worker["rank"],
                        "gpu": worker["gpu"],
                        "versions": worker["versions"],
                        "loaded_module_sha256": {
                            name: source["sha256"]
                            for name, source in worker["sources"].items()
                        },
                    }
                    for worker in sorted(workers, key=lambda item: item["rank"])
                ],
                "mean_ms": statistics.mean(timings),
                "median_ms": statistics.median(timings),
                "sample_sd_ms": statistics.stdev(timings),
                "encoder_cuda_ms_by_rank": {
                    str(audit["rank"]): sum(
                        call["cuda_ms"] for call in audit["encoder_calls"]
                    )
                    for audit in audits
                },
                "untimed_encoder_calls": audits,
                "worker_backend_counts": [
                    dict(
                        Counter(
                            (p["backend"] or p["method"])
                            for p in worker["projections"]
                            if p["name"].startswith("text_encoder")
                        )
                    )
                    for worker in workers
                ],
                "encoder_native_linear_counts": [
                    len(worker["encoder_native_linears"]) for worker in workers
                ],
                "dit_bf16_projection_counts": [
                    sum(
                        p["name"].startswith("transformer.")
                        for p in worker["projections"]
                    )
                    for worker in workers
                ],
                "vae_parameter_dtypes": [worker["vae_dtypes"] for worker in workers],
                "untimed_encoder_activation_dtypes": [
                    audit["encoder_activation_dtypes"] for audit in audits
                ],
                "peak_reserved_mib_by_rank": {
                    str(worker["rank"]): worker["peak_reserved_bytes"] / 2**20
                    for worker in workers
                },
                "peak_reserved_mib": max(
                    worker["peak_reserved_bytes"] for worker in workers
                )
                / 2**20,
            }
        )
    groups = defaultdict(list)
    for row in rows:
        groups[row["resolution"], row["variant"], row["hsdp"]].append(row)
    assert len(groups) == 12 and all(len(group) == 2 for group in groups.values())
    runtime = rows[0]["runtime_by_rank"]
    assert all(row["runtime_by_rank"] == runtime for row in rows)
    assert runtime[0]["versions"] == runtime[1]["versions"]
    summary = {
        "source": plan["validation"]["source"],
        "sm80_quality_gate": gate,
        "rows": rows,
        "timed_requests": 144,
        "physical_gpu_count": 2,
        "boot_id": plan["boot_id"],
        "gpu_uuids": plan["gpu_uuids"],
        "runtime_by_rank": runtime,
        "model": {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "verified_files": len(model["files"]),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "limits": "Lifecycle audits passed; manual image and native backend reviews remain required. Peak is worker-lifetime reserved memory per rank, including untimed audits, not total model residency or a timing-phase-only peak.",
    }
    lines = [
        "# A100 Ovis dual-GPU complete-request validation",
        "",
        "Two fresh processes per row; six timed requests per process, warmups excluded.",
        "SM80 encoder FP8 uses the audited weight-only fallback with BF16 activations; native FP8 GEMM is not available on A100.",
        "Memory reports each worker's lifetime peak reserved memory, including untimed profiling/lifecycle audits; it is not a timing-phase-only peak.",
        "",
        "| Resolution HxW | Precision | HSDP | Mean ms | Median ms | Sample SD ms | Process means ms | Peak reserved R0/R1 MiB |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for (resolution, variant, mode), group in groups.items():
        means = [row["mean_ms"] for row in group]
        samples = [value for row in group for value in row["timed_ms"]]
        peaks = [
            max(row["peak_reserved_mib_by_rank"][str(rank)] for row in group)
            for rank in range(2)
        ]
        lines.append(
            f"| {resolution} | {variant} | {mode} | {statistics.mean(means):.2f} | "
            f"{statistics.median(samples):.2f} | {statistics.stdev(samples):.2f} | "
            f"{', '.join(f'{value:.2f}' for value in means)} | "
            f"{peaks[0]:.0f}/{peaks[1]:.0f} |"
        )
    lines += [
        "",
        "Untimed encoder CUDA events: sum of positive and negative forward GPU durations, profiled separately on each rank; tokenization and input preparation excluded.",
        "",
        "| Resolution HxW | Precision | HSDP | Encoder CUDA mean R0/R1 ms |",
        "|---|---|---|---|",
    ]
    for (resolution, variant, mode), group in groups.items():
        encoder = [
            statistics.mean(row["encoder_cuda_ms_by_rank"][str(rank)] for row in group)
            for rank in range(2)
        ]
        lines.append(
            f"| {resolution} | {variant} | {mode} | {encoder[0]:.3f}/{encoder[1]:.3f} |"
        )
    lines += [
        "",
        "Matched comparisons; negative reductions mean slower execution.",
        "",
        "| Resolution HxW | Comparison | Request latency reduction |",
        "|---|---|---:|",
    ]
    averages = {
        key: statistics.mean(row["mean_ms"] for row in group)
        for key, group in groups.items()
    }
    for resolution in dict.fromkeys(row["resolution"] for row in rows):
        for mode in ("none", "block", "resident"):
            reduction = (
                1
                - averages[resolution, "encoder", mode]
                / averages[resolution, "bf16", mode]
            ) * 100
            lines.append(
                f"| {resolution} | Encoder FP8 weights / BF16 activations vs BF16, {mode} | {reduction:.2f}% |"
            )
        for variant in ("bf16", "encoder"):
            reduction = (
                1
                - averages[resolution, variant, "resident"]
                / averages[resolution, variant, "block"]
            ) * 100
            lines.append(
                f"| {resolution} | Resident vs block HSDP, {variant} | {reduction:.2f}% |"
            )
    (root / "ovis/evidence/a100-ovis-full-v9-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (root / "ovis/evidence/a100-ovis-full-v9-speed-table.md").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
