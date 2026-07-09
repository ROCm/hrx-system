# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

from loom.tools.kernel_anatomy import (
    ComparisonSpec,
    NamedPath,
    build_kernel_anatomy_comparisons,
    build_kernel_anatomy_report,
    format_text_report,
    main,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_compile_report(path: Path) -> None:
    _write(
        path,
        json.dumps(
            {
                "compile_report": _compile_report_payload(),
            }
        ),
    )


def _compile_report_payload() -> dict:
    return {
        "function": "loom_kernel",
        "target_key": "gfx1100",
        "executable_format": "amdgcn-amd-amdhsa--gfx1100",
        "entries": {
            "rows": [
                {
                    "function": "loom_kernel",
                    "instruction_count": 123,
                    "code_byte_count": 456,
                    "private_memory_bytes": 0,
                    "local_memory_bytes": 2048,
                    "static_instruction_mix": {
                        "vector_alu_count": 17,
                        "wmma_count": 2,
                        "mfma_count": 0,
                    },
                    "dynamic_instruction_mix": {
                        "local_memory_count": 99,
                    },
                    "target_resources": {
                        "vector": {
                            "final": {
                                "register_count": 64,
                            },
                        },
                        "occupancy_percent": 50,
                    },
                    "workload": {
                        "workgroup_size": {
                            "flat": 128,
                        },
                    },
                },
            ],
        },
    }


def _write_benchmark_jsonl(path: Path) -> None:
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "benchmark",
                        "candidate_id": "c0",
                        "candidate_index": 0,
                        "benchmark_result": {
                            "benchmark": "bench_kernel",
                            "case": "case_kernel",
                            "state": "ok",
                            "sample_compilation": "once",
                            "correctness": {
                                "sample_count": 1,
                                "failed_sample_count": 0,
                            },
                            "measurement": {
                                "operation_timing_ns": {
                                    "count": 5,
                                    "p50": 1234000,
                                    "p90": 1250000,
                                },
                                "timing_interpretation": {
                                    "warnings": ["low_sample_count"],
                                },
                            },
                            "compile_report": _compile_report_payload(),
                        },
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_rocblas_log(path: Path) -> None:
    _write(
        path,
        """
rocBLAS version: 5.5.0.example
rocBLAS-commit-hash: abcdef
Tensile-commit-hash: 123456
hipBLASLt version: 1.4.0 commit-hash: cafe
Device ID 0 : AMD Radeon Pro W7900 Dual Slot gfx1100
Library logic solution index of winning solution: 44
Running kernel: selected_kernel
Kernel name: selected_kernel
Kernel parameters:
           MatrixInstruction: (16, 16, 16, 1)
               workGroupSize: (32, 4, 1)
                   macroTile: (128, 128, 1)
                      depthU: 16

transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,ldd,batch_count,cold_iters,hot_iters,rocblas-Gflops,us
T,N,13824,4547,4608,1,4608,0,4608,13824,13824,1,5,20,68121,8503.93
""",
    )


def test_build_kernel_anatomy_report_merges_disassembly_and_compile_report(
    tmp_path: Path,
) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <small>:
      global_load_b128 v[0:3], v0
0000000000000100 <big>:
      global_load_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        top_symbol_count=1,
        ordered_symbol_count=2,
    )

    assert report["schema"] == "loom.kernel_anatomy"
    assert report["schema_version"] == 1
    assert report["loom_compile_reports"]["loom"]["function"] == "loom_kernel"
    assert report["loom_compile_reports"]["loom"]["local_memory_bytes"] == 2048
    assert report["disassemblies"]["asm"]["whole_file"]["family_counts"]["v_wmma"] == 2
    assert report["disassemblies"]["asm"]["top_symbols"][0]["symbol"] == "big"
    ordered_symbols = report["disassemblies"]["asm"]["ordered_symbols"]
    assert [symbol["symbol"] for symbol in ordered_symbols] == ["small", "big"]
    assert [symbol["address"] for symbol in ordered_symbols] == [0, 0x100]


def test_build_kernel_anatomy_report_extracts_benchmark_jsonl(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "results.jsonl"
    _write_benchmark_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )

    benchmark_report = report["loom_benchmarks"]["bench"]
    assert benchmark_report["benchmark_count"] == 1
    benchmark = benchmark_report["benchmarks"][0]
    assert benchmark["benchmark"] == "bench_kernel"
    assert benchmark["timing_ns"]["p50"] == 1234000
    assert benchmark["correctness"]["failed_sample_count"] == 0
    assert benchmark["compile_report"]["instruction_count"] == 123
    assert (
        benchmark["compile_report"]["dynamic_instruction_mix"]["local_memory_count"]
        == 99
    )


def test_build_kernel_anatomy_report_extracts_rocblas_log(
    tmp_path: Path,
) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    _write_rocblas_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("tensile", rocblas_log_path)],
    )

    rocblas_log = report["rocblas_logs"]["tensile"]
    assert rocblas_log["rocblas_version"] == "5.5.0.example"
    assert rocblas_log["solution_index"] == 44
    assert rocblas_log["running_kernel"] == "selected_kernel"
    assert rocblas_log["devices"][0]["arch"] == "gfx1100"
    assert rocblas_log["kernel_parameters"]["macroTile"] == "(128, 128, 1)"
    assert rocblas_log["timing_rows"][0]["M"] == 13824
    assert rocblas_log["timing_rows"][0]["us"] == 8503.93


def test_symbol_regex_filters_disassembly_blocks(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <skip_me>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000100 <keep_me>:
      global_store_b128 v0, v[0:3]
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[],
        symbol_regex="keep",
        top_symbol_count=8,
    )

    symbols = report["disassemblies"]["asm"]["top_symbols"]
    assert [symbol["symbol"] for symbol in symbols] == ["keep_me"]


def test_ordered_symbols_preserve_disassembly_address_order(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000100 <middle>:
      ds_read_b64 v[0:1], v0
0000000000000000 <first>:
      global_load_b128 v[0:3], v0
0000000000000200 <last>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[],
        ordered_symbol_count=2,
    )

    ordered_symbols = report["disassemblies"]["asm"]["ordered_symbols"]
    assert [symbol["symbol"] for symbol in ordered_symbols] == ["first", "middle"]
    assert [symbol["address"] for symbol in ordered_symbols] == [0, 0x100]


def test_amdhsa_metadata_extracts_kernel_resources(tmp_path: Path) -> None:
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
      - .name:           Tensor2dSizeA
        .offset:         0
        .size:           8
    .group_segment_fixed_size: 25600
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .max_flat_workgroup_size: 128
    .name:           selected_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         selected_kernel.kd
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .name:           Tensor2dSizeA
        .offset:         0
        .size:           8
    .group_segment_fixed_size: 4096
    .name:           skipped_kernel
    .symbol:         skipped_kernel.kd
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        amdhsa_metadata_paths=[NamedPath("hsaco", metadata_path)],
        amdhsa_metadata_regex="selected",
    )

    metadata = report["amdhsa_metadata"]["hsaco"]
    assert metadata["kernel_count"] == 1
    assert metadata["kernels"][0]["symbol"] == "selected_kernel.kd"
    assert metadata["kernels"][0]["group_segment_fixed_size"] == 25600
    assert metadata["kernels"][0]["vgpr_count"] == 256


def test_text_report_contains_kernel_economics(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <kernel>:
      global_load_b128 v[0:3], v0
      global_store_b128 v0, v[0:3]
      buffer_load_b64 v[0:1], v0, s[0:3], 0 offen
      buffer_store_b64 v[0:1], v0, s[0:3], 0 offen
      s_waitcnt vmcnt(0)
      s_barrier
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
      - .name:           Tensor2dSizeA
        .offset:         0
    .group_segment_fixed_size: 25600
    .kernarg_segment_size: 128
    .max_flat_workgroup_size: 128
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         kernel.kd
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        amdhsa_metadata_paths=[NamedPath("hsaco", metadata_path)],
        ordered_symbol_count=1,
    )
    text = format_text_report(report)

    assert "Kernel anatomy report" in text
    assert "loom: instructions=123 code_bytes=456 local_bytes=2048" in text
    assert "asm: symbols=1 instructions=6" in text
    assert "global_load=1 global_store=1 buffer_load=1 buffer_store=1" in text
    assert "wait=1 barrier=1 read_bytes=24 write_bytes=24" in text
    assert "ordered symbols:" in text
    assert "0x0 kernel: instructions=6" in text
    assert "hsaco: kernels=1" in text
    assert "kernel.kd: lds=25600 private=0 vgpr=256 sgpr=66" in text


def test_text_report_contains_benchmark_jsonl_summary(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "results.jsonl"
    _write_benchmark_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )
    text = format_text_report(report)

    assert "Loom benchmark JSONL:" in text
    assert "bench: benchmarks=1" in text
    assert "bench_kernel: state=ok p50_ms=1.234" in text
    assert "correctness=0/1 instructions=123 code_bytes=456" in text
    assert "local_bytes=2048 vgpr=64 occupancy=50%" in text
    assert "wmma=2 valu=17 dynamic_local=99" in text


def test_text_report_contains_rocblas_log_summary(tmp_path: Path) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    _write_rocblas_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("tensile", rocblas_log_path)],
    )
    text = format_text_report(report)

    assert "rocBLAS logs:" in text
    assert "tensile: solution=44 arch=gfx1100 kernel=selected_kernel" in text
    assert "M=13824 N=4547 K=4608 time_ms=8.50393 gflops=68121" in text
    assert "MatrixInstruction: (16, 16, 16, 1)" in text


def test_build_kernel_anatomy_comparisons_ranks_mixed_artifact_metrics(
    tmp_path: Path,
) -> None:
    disassembly_path = tmp_path / "tensile.s"
    _write(
        disassembly_path,
        """
0000000000000000 <tensile_kernel>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
      v_add_f32 v0, v1, v2
""",
    )
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
    .group_segment_fixed_size: 1024
    .max_flat_workgroup_size: 128
    .private_segment_fixed_size: 0
    .symbol: tensile_kernel.kd
    .vgpr_count: 32
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)
    benchmark_path = tmp_path / "results.jsonl"
    _write_benchmark_jsonl(benchmark_path)
    rocblas_log_path = tmp_path / "rocblas.log"
    _write_rocblas_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("tensile", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        benchmark_jsonl_paths=[NamedPath("loom", benchmark_path)],
        rocblas_log_paths=[NamedPath("tensile", rocblas_log_path)],
        amdhsa_metadata_paths=[NamedPath("tensile", metadata_path)],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report, [ComparisonSpec(baseline="tensile", candidate="loom")]
    )

    comparison = comparisons["tensile=loom"]
    assert comparison["shared_metric_count"] >= 4
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["local_memory_bytes"]["baseline"] == 1024
    assert deltas_by_metric["local_memory_bytes"]["candidate"] == 2048
    assert deltas_by_metric["local_memory_bytes"]["ratio"] == 2
    assert deltas_by_metric["wmma_count"]["baseline"] == 1
    assert deltas_by_metric["wmma_count"]["candidate"] == 2
    assert deltas_by_metric["vgpr_count"]["baseline"] == 32
    assert deltas_by_metric["vgpr_count"]["candidate"] == 64
    assert deltas_by_metric["operation_time_ns"]["baseline"] == 8503930.0
    assert deltas_by_metric["operation_time_ns"]["candidate"] == 1234000


def test_text_report_contains_comparison_summary(tmp_path: Path) -> None:
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
    .group_segment_fixed_size: 1024
    .symbol: baseline_kernel.kd
    .vgpr_count: 32
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        amdhsa_metadata_paths=[NamedPath("baseline", metadata_path)],
    )
    report["comparisons"] = build_kernel_anatomy_comparisons(
        report, [ComparisonSpec(baseline="baseline", candidate="loom")]
    )
    text = format_text_report(report)

    assert "Comparisons:" in text
    assert "baseline=loom: shared=2 baseline_metrics=2" in text
    assert "local_memory_bytes: baseline=1024 candidate=2048" in text
    assert "ratio=2x sources=amdhsa_metadata/compile_report" in text


def test_main_emits_json(tmp_path: Path, capsys) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <kernel>:
      s_endpgm
""",
    )

    assert main(["--disassembly", f"asm={disassembly_path}", "--format", "json"]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["schema"] == "loom.kernel_anatomy"
    assert report["disassemblies"]["asm"]["symbol_count"] == 1


def test_main_emits_comparisons(tmp_path: Path, capsys) -> None:
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)
    benchmark_path = tmp_path / "results.jsonl"
    _write_benchmark_jsonl(benchmark_path)
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
    .group_segment_fixed_size: 1024
    .symbol: baseline_kernel.kd
    .vgpr_count: 32
""",
    )
    rocblas_log_path = tmp_path / "rocblas.log"
    _write_rocblas_log(rocblas_log_path)

    assert (
        main(
            [
                "--compile-report",
                f"loom={compile_report_path}",
                "--benchmark-jsonl",
                f"loom={benchmark_path}",
                "--amdhsa-metadata",
                f"baseline={metadata_path}",
                "--rocblas-log",
                f"baseline={rocblas_log_path}",
                "--compare",
                "baseline=loom",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    comparison = report["comparisons"]["baseline=loom"]
    assert comparison["shared_metric_count"] == 3
