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
    SymbolWeightSpec,
    build_kernel_anatomy_comparison_scorecard,
    build_kernel_anatomy_comparisons,
    build_kernel_anatomy_report,
    format_rocblas_replay_script,
    format_text_report,
    main,
)

_TENSILE_SYMBOL = (
    "Cijk_Alik_Bljk_BBS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_AMAS3_BL1_BS1_"
    "EPS1_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1100_IU1_K1_KLA_LBSPPA128_"
    "LBSPPB128_LPA8_LPB8_LRVW16_MIAV1_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA3_SS1_"
    "SU0_SUM0_SUS0_SVW4_TT4_64_TLDS1_UMLDSA1_UMLDSB1_USFGROn1_VAW1_VSn1_"
    "VW4_VWB2_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4"
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
        "economics": {
            "memory": {
                "source_low": {
                    "packet_count": 560,
                    "load_packet_count": 288,
                    "store_packet_count": 272,
                    "dynamic_packet_count": 78464,
                    "dynamic_read_byte_count": 221184,
                    "dynamic_write_byte_count": 147712,
                    "dispatch_source": {
                        "read_bytes": 110075314176,
                        "write_bytes": 73510944768,
                        "total_bytes": 183586258944,
                    },
                    "argument_packets": {
                        "count": 3,
                        "rows": [
                            {
                                "root_argument_name": "input",
                                "memory_space": "global",
                                "operation": "load",
                                "packet": "amdgpu.global_load_b128_saddr",
                                "packet_count": 16,
                                "dynamic_packet_count": 2304,
                                "dispatch_source": {
                                    "read_bytes": 18345885696,
                                    "total_bytes": 18345885696,
                                },
                            },
                            {
                                "root_argument_name": "weight",
                                "memory_space": "global",
                                "operation": "load",
                                "packet": "amdgpu.global_load_b128_saddr",
                                "packet_count": 16,
                                "dynamic_packet_count": 2304,
                                "dispatch_source": {
                                    "read_bytes": 18345885696,
                                    "total_bytes": 18345885696,
                                },
                            },
                            {
                                "root_argument_name": "output",
                                "memory_space": "global",
                                "operation": "store",
                                "packet": "amdgpu.global_store_b16_saddr",
                                "packet_count": 16,
                                "dynamic_packet_count": 128,
                                "dispatch_source": {
                                    "write_bytes": 127401984,
                                    "total_bytes": 127401984,
                                },
                            },
                        ],
                    },
                },
            },
        },
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
                        "matrix_count": 20,
                        "wmma_count": 20,
                        "mfma_count": 0,
                        "local_memory_count": 99,
                        "local_read_byte_count": 320,
                        "local_write_byte_count": 160,
                    },
                    "target_resources": {
                        "vector": {
                            "final": {
                                "register_count": 64,
                            },
                        },
                        "occupancy_percent": 50,
                    },
                    "wait_plan": {
                        "action_count": 138,
                        "full_drain_count": 27,
                        "partial_wait_count": 111,
                        "drained_count": 417,
                        "max_outstanding_before": 128,
                    },
                    "wait_reason_summary_rows": {
                        "count": 2,
                        "rows": [
                            {
                                "counter": "lds",
                                "reason": "amdgpu.read_result_reuse",
                                "summary": {
                                    "action_count": 120,
                                    "full_drain_count": 11,
                                    "partial_wait_count": 109,
                                    "drained_count": 120,
                                    "max_outstanding_before": 32,
                                },
                            },
                            {
                                "counter": "vmem_load",
                                "reason": "amdgpu.ssa_use",
                                "summary": {
                                    "action_count": 8,
                                    "full_drain_count": 7,
                                    "partial_wait_count": 1,
                                    "drained_count": 17,
                                    "max_outstanding_before": 4,
                                },
                            },
                        ],
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
        f"""
rocBLAS version: 5.5.0.example
rocBLAS-commit-hash: abcdef
Tensile-commit-hash: 123456
hipBLASLt version: 1.4.0 commit-hash: cafe
Device ID 0 : AMD Radeon Pro W7900 Dual Slot gfx1100
Library logic solution index of winning solution: 44
Running kernel: {_TENSILE_SYMBOL}
Kernel name: {_TENSILE_SYMBOL}
Kernel parameters:
           MatrixInstruction: (16, 16, 16, 1)
               workGroupSize: (32, 4, 1)
                   macroTile: (128, 128, 1)
                      depthU: 16

transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,ldd,batch_count,cold_iters,hot_iters,rocblas-Gflops,us
T,N,13824,4547,4608,1,4608,0,4608,13824,13824,1,5,20,68121,8503.93
with 48.3 GB memory, max. SCLK 1760 MHz, max. MCLK 1124 MHz, memoryBusWidth 48 Bytes, compute capability 11.0
""",
    )


def _write_rocblas_trace_log(path: Path) -> None:
    _write(
        path,
        """
- { rocblas_function: "rocblas_gemm_ex", atomics_mode: atomics_allowed, a_type: "bf16_r", b_type: "bf16_r", c_type: "bf16_r", d_type: "bf16_r", compute_type: "f32_r", transA: 'T', transB: 'N', M: 12288, N: 4547, K: 4608, alpha: 1.0, lda: 4608, beta: 0.0, ldb: 4608, ldc: 12288, ldd: 12288, batch_count: 1, algo: 0, solution_index: 0, flags: none, call_count: 136 }
- { rocblas_function: "rocblas_gemm_ex", atomics_mode: atomics_not_allowed, a_type: "bf16_r", b_type: "bf16_r", c_type: "bf16_r", d_type: "bf16_r", compute_type: "f32_r", transA: 'T', transB: 'N', M: 13824, N: 4547, K: 4608, alpha: 1.0, lda: 4608, beta: 0.0, ldb: 4608, ldc: 13824, ldd: 13824, batch_count: 1, algo: 0, solution_index: 0, flags: none, cold_iters: 5, iters: 20, device: 1, call_count: 68 }
""",
    )


def _write_iree_dispatch_profile(path: Path) -> None:
    _write(
        path,
        json.dumps(
            {
                "by_kernel": [
                    {
                        "count": 3,
                        "function_name": "cold_kernel",
                        "invalid_count": 0,
                        "max_duration_ns": 3000,
                        "min_duration_ns": 1000,
                        "module_path": "model/cold",
                        "p50_duration_ns": 2000,
                        "p90_duration_ns": 3000,
                        "p99_duration_ns": 3000,
                        "total_duration_ns": 6000,
                        "valid_count": 3,
                    },
                    {
                        "count": 2,
                        "function_name": "hot_kernel",
                        "invalid_count": 0,
                        "max_duration_ns": 11000,
                        "min_duration_ns": 9000,
                        "module_path": "model/hot",
                        "p50_duration_ns": 10000,
                        "p90_duration_ns": 11000,
                        "p99_duration_ns": 11000,
                        "total_duration_ns": 20000,
                        "valid_count": 2,
                    },
                ],
            }
        ),
    )


def _write_rocprof_kernel_trace(path: Path) -> None:
    _write(
        path,
        """"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH","Agent 5",1,0,10,1,9,"hot_kernel",1,1000,1100,2048,0,64,0,80,64,2,1,16,8,1
"KERNEL_DISPATCH","Agent 5",1,0,10,2,9,"hot_kernel",2,2000,2200,2048,0,64,0,80,64,2,1,16,8,1
"KERNEL_DISPATCH","Agent 5",1,0,10,3,9,"hot_kernel",3,3000,3300,2048,0,64,0,80,64,2,1,16,8,1
"KERNEL_DISPATCH","Agent 5",1,0,10,4,8,"cold_kernel",4,4000,4050,0,0,16,0,48,32,1,1,8,1,1
""",
    )


def _write_rocprof_counter_collection(path: Path) -> None:
    _write(
        path,
        """"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
1,1,"Agent 5",1,20,20,128,9,"hot_kernel",128,2048,0,64,0,80,"SQ_INSTS_LDS",10,1000,1100
1,1,"Agent 5",1,20,20,128,9,"hot_kernel",128,2048,0,64,0,80,"LDSBankConflict",2,1000,1100
2,2,"Agent 5",1,20,20,128,9,"hot_kernel",128,2048,0,64,0,80,"SQ_INSTS_LDS",30,2000,2200
2,2,"Agent 5",1,20,20,128,9,"hot_kernel",128,2048,0,64,0,80,"LDSBankConflict",4,2000,2200
3,3,"Agent 5",1,20,20,8,8,"cold_kernel",32,0,0,16,0,48,"SQ_INSTS_LDS",1,3000,3050
3,3,"Agent 5",1,20,20,8,8,"cold_kernel",32,0,0,16,0,48,"LDSBankConflict",0,3000,3050
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
    assert report["loom_compile_reports"]["loom"]["wait_plan"]["action_count"] == 138
    assert (
        report["loom_compile_reports"]["loom"]["wait_reasons"][0]["reason"]
        == "amdgpu.read_result_reuse"
    )
    assert (
        report["loom_compile_reports"]["loom"]["wait_reasons"][0]["partial_wait_count"]
        == 109
    )
    source_low = report["loom_compile_reports"]["loom"]["source_low"]
    assert source_low["dynamic_packet_count"] == 78464
    assert source_low["dispatch_source"]["total_bytes"] == 183586258944
    assert source_low["argument_packets"][0]["root_argument_name"] == "input"
    assert (
        source_low["argument_packets"][0]["packet"] == "amdgpu.global_load_b128_saddr"
    )
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
    assert rocblas_log["running_kernel"] == _TENSILE_SYMBOL
    assert rocblas_log["devices"][0]["arch"] == "gfx1100"
    assert rocblas_log["kernel_parameters"]["macroTile"] == "(128, 128, 1)"
    symbol_parameters = rocblas_log["symbol_parameters"]
    assert symbol_parameters["macro_tile"] == {"x": 128, "y": 128, "z": 16}
    assert symbol_parameters["matrix_instruction"] == {
        "x": 16,
        "y": 16,
        "z": 16,
        "w": 1,
    }
    assert symbol_parameters["workgroup_size"] == {"x": 32, "y": 4, "z": 1}
    assert symbol_parameters["flat_workgroup_size"] == 128
    assert symbol_parameters["thread_tile"] == {"x": 4, "y": 64}
    assert symbol_parameters["wave_size"] == 32
    assert symbol_parameters["global_load_vector_width_a"] == 4
    assert symbol_parameters["global_load_vector_width_b"] == 4
    assert symbol_parameters["local_read_vector_width"] == 16
    assert len(rocblas_log["timing_rows"]) == 1
    assert rocblas_log["timing_rows"][0]["M"] == 13824
    assert rocblas_log["timing_rows"][0]["us"] == 8503.93


def test_build_kernel_anatomy_report_extracts_rocblas_trace_rows(
    tmp_path: Path,
) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("pytorch", rocblas_log_path)],
    )

    rocblas_log = report["rocblas_logs"]["pytorch"]
    assert len(rocblas_log["trace_rows"]) == 2
    assert rocblas_log["trace_rows"][0]["rocblas_function"] == "rocblas_gemm_ex"
    assert rocblas_log["trace_rows"][0]["M"] == 12288
    assert rocblas_log["trace_rows"][0]["transA"] == "T"
    assert rocblas_log["trace_rows"][0]["call_count"] == 136
    assert rocblas_log["trace_rows"][0]["operation_key"] == "rocblas_gemm_ex"
    assert rocblas_log["trace_rows"][0]["shape_key"] == "M12288_N4547_K4608_beta0.0"
    assert rocblas_log["trace_rows"][0]["rocblas_bench_arguments"] == [
        "--function",
        "gemm_ex",
        "--transposeA",
        "T",
        "--transposeB",
        "N",
        "--sizem",
        "12288",
        "--sizen",
        "4547",
        "--sizek",
        "4608",
        "--a_type",
        "bf16_r",
        "--b_type",
        "bf16_r",
        "--c_type",
        "bf16_r",
        "--d_type",
        "bf16_r",
        "--compute_type",
        "f32_r",
        "--alpha",
        "1",
        "--beta",
        "0",
        "--lda",
        "4608",
        "--ldb",
        "4608",
        "--ldc",
        "12288",
        "--ldd",
        "12288",
        "--batch_count",
        "1",
        "--algo",
        "0",
        "--solution_index",
        "0",
        "--flags",
        "0",
        "--atomics_allowed",
    ]
    assert "--cold_iters" not in rocblas_log["trace_rows"][0]["rocblas_bench_arguments"]
    assert (
        "--atomics_not_allowed"
        in rocblas_log["trace_rows"][1]["rocblas_bench_arguments"]
    )
    assert "--cold_iters" in rocblas_log["trace_rows"][1]["rocblas_bench_arguments"]
    assert "--iters" in rocblas_log["trace_rows"][1]["rocblas_bench_arguments"]
    assert "--device" in rocblas_log["trace_rows"][1]["rocblas_bench_arguments"]


def test_build_kernel_anatomy_report_extracts_iree_dispatch_profile(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "dispatch_profile.json"
    _write_iree_dispatch_profile(profile_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        iree_dispatch_profile_paths=[NamedPath("stage", profile_path)],
    )

    profile = report["iree_dispatch_profiles"]["stage"]
    assert profile["kernel_count"] == 2
    assert profile["kernels"][0]["function_name"] == "hot_kernel"
    assert profile["kernels"][0]["total_duration_ns"] == 20000
    assert profile["kernels"][1]["function_name"] == "cold_kernel"


def test_build_kernel_anatomy_report_extracts_rocprof_kernel_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "kernel_trace.csv"
    _write_rocprof_kernel_trace(trace_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocprof_kernel_trace_paths=[NamedPath("profile", trace_path)],
    )

    trace = report["rocprof_kernel_traces"]["profile"]
    assert trace["dispatch_count"] == 4
    assert trace["kernel_count"] == 2
    hot_kernel = trace["kernels"][0]
    assert hot_kernel["kernel_name"] == "hot_kernel"
    assert hot_kernel["count"] == 3
    assert hot_kernel["total_duration_ns"] == 600
    assert hot_kernel["p50_duration_ns"] == 200
    assert hot_kernel["local_memory_bytes"] == 2048
    assert hot_kernel["workgroup_size"] == 128
    assert hot_kernel["grid_size"] == 128


def test_build_kernel_anatomy_report_extracts_rocprof_counters(
    tmp_path: Path,
) -> None:
    counter_path = tmp_path / "counter_collection.csv"
    _write_rocprof_counter_collection(counter_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocprof_counter_collection_paths=[NamedPath("profile", counter_path)],
    )

    collection = report["rocprof_counter_collections"]["profile"]
    assert collection["row_count"] == 6
    assert collection["dispatch_count"] == 3
    assert collection["counter_count"] == 2
    hot_kernel = collection["kernels"][0]
    assert hot_kernel["kernel_name"] == "hot_kernel"
    assert hot_kernel["count"] == 2
    assert hot_kernel["total_duration_ns"] == 300
    assert hot_kernel["counters"]["SQ_INSTS_LDS"]["sum"] == 40
    assert hot_kernel["counters"]["SQ_INSTS_LDS"]["mean"] == 20
    assert hot_kernel["counters"]["LDSBankConflict"]["max"] == 4


def test_rocprof_inputs_with_matching_names_merge_as_passes(tmp_path: Path) -> None:
    trace_path_0 = tmp_path / "pass_0" / "kernel_trace.csv"
    trace_path_1 = tmp_path / "pass_1" / "kernel_trace.csv"
    counter_path_0 = tmp_path / "pass_0" / "counter_collection.csv"
    counter_path_1 = tmp_path / "pass_1" / "counter_collection.csv"
    _write_rocprof_kernel_trace(trace_path_0)
    _write_rocprof_kernel_trace(trace_path_1)
    _write_rocprof_counter_collection(counter_path_0)
    _write_rocprof_counter_collection(counter_path_1)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocprof_kernel_trace_paths=[
            NamedPath("profile", trace_path_0),
            NamedPath("profile", trace_path_1),
        ],
        rocprof_counter_collection_paths=[
            NamedPath("profile", counter_path_0),
            NamedPath("profile", counter_path_1),
        ],
    )

    trace = report["rocprof_kernel_traces"]["profile"]
    assert trace["path"] is None
    assert trace["dispatch_count"] == 8
    assert trace["kernels"][0]["count"] == 6
    collection = report["rocprof_counter_collections"]["profile"]
    assert collection["path"] is None
    assert collection["row_count"] == 12
    assert collection["dispatch_count"] == 6
    hot_kernel = collection["kernels"][0]
    assert hot_kernel["count"] == 4
    assert hot_kernel["counters"]["SQ_INSTS_LDS"]["sum"] == 80


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


def test_symbol_weights_estimate_dynamic_block_counts(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <prologue>:
      buffer_load_b128 v[0:3], v0, s[0:3], 0 offen
0000000000000100 <open_loop>:
      buffer_load_b128 v[0:3], v0, s[0:3], 0 offen
      ds_write_b128 v0, v[0:3]
      ds_read_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000200 <tail>:
      buffer_store_b128 v0, v[0:3], s[0:3], 0 offen
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("tensile", disassembly_path)],
        compile_report_paths=[],
        symbol_weight_specs=[
            SymbolWeightSpec("tensile", "open_loop", 7),
            SymbolWeightSpec("tensile", "tail", 1),
        ],
    )

    weighted_symbols = report["disassemblies"]["tensile"]["weighted_symbols"]
    assert weighted_symbols["matched_symbol_count"] == 2
    assert weighted_symbols["rules"][0]["matched_symbols"] == ["open_loop"]
    summary = weighted_symbols["summary"]
    assert summary["instruction_count"] == 29
    assert summary["family_counts"]["buffer_load"] == 7
    assert summary["family_counts"]["buffer_store"] == 1
    assert summary["family_counts"]["ds_read"] == 7
    assert summary["family_counts"]["ds_write"] == 7
    assert summary["family_counts"]["v_wmma"] == 7
    assert summary["memory_byte_counts"]["read_bytes"] == 224
    assert summary["memory_byte_counts"]["write_bytes"] == 128


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
    assert "waits: actions=138 full=27 partial=111 drained=417" in text
    assert "wait lds/amdgpu.read_result_reuse: actions=120" in text
    assert "source-low: packets=560 dynamic_packets=78464" in text
    assert "dispatch_total_bytes=183586258944" in text
    assert "input/load: packet=amdgpu.global_load_b128_saddr" in text
    assert "asm: symbols=1 instructions=6" in text
    assert "global_load=1 global_store=1 buffer_load=1 buffer_store=1" in text
    assert "wait=1 barrier=1 read_bytes=24 write_bytes=24" in text
    assert "ordered symbols:" in text
    assert "0x0 kernel: instructions=6" in text
    assert "hsaco: kernels=1" in text
    assert "kernel.kd: lds=25600 private=0 vgpr=256 sgpr=66" in text


def test_text_report_contains_weighted_symbols(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <open_loop>:
      ds_read_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[],
        symbol_weight_specs=[SymbolWeightSpec("asm", "open_loop", 4)],
    )
    text = format_text_report(report)

    assert "weighted symbols: rules=1 matches=1 instructions=8" in text
    assert "wmma=4" in text
    assert "ds_read=4" in text
    assert "open_loop: weight=4 matches=1 symbols=open_loop" in text


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
    assert f"tensile: solution=44 arch=gfx1100 kernel={_TENSILE_SYMBOL}" in text
    assert "symbol tile: MT=128x128x16 MI=16x16x16x1 WG=32x4x1" in text
    assert "TT=4x64 WS=32 VW=4 GLVWA=4 GLVWB=4 LRVW=16" in text
    assert "M=13824 N=4547 K=4608 time_ms=8.50393 gflops=68121" in text
    assert "MatrixInstruction: (16, 16, 16, 1)" in text


def test_text_report_contains_rocblas_trace_rows(tmp_path: Path) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("pytorch", rocblas_log_path)],
    )
    text = format_text_report(report)

    assert "trace rows:" in text
    assert "M=12288 N=4547 K=4608 beta=0.0 solution=0 calls=136" in text
    assert "M=13824 N=4547 K=4608 beta=0.0 solution=0 calls=68" in text
    assert "replay: rocblas-bench --function gemm_ex" in text
    assert "--sizem 12288 --sizen 4547 --sizek 4608" in text
    assert "--flags 0 --atomics_allowed" in text
    assert "--cold_iters 5 --iters 20 --device 1 --atomics_not_allowed" in text


def test_rocblas_replay_script_contains_profile_commands(tmp_path: Path) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("pytorch", rocblas_log_path)],
    )
    text = format_rocblas_replay_script(
        report,
        executable="/opt/rocm/bin/rocblas-bench",
        extra_arguments=["--cold_iters", "7", "--iters", "11"],
    )

    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "# pytorch M12288_N4547_K4608_beta0.0 calls=136" in text
    assert "/opt/rocm/bin/rocblas-bench --function gemm_ex" in text
    assert "--sizem 12288 --sizen 4547 --sizek 4608" in text
    assert "--flags 0 --atomics_allowed" in text
    assert "# pytorch M13824_N4547_K4608_beta0.0 calls=68" in text
    assert "--cold_iters 5 --iters 20 --device 1 --atomics_not_allowed" in text
    assert "--atomics_allowed --cold_iters 7 --iters 11" in text


def test_text_report_contains_iree_dispatch_profile_summary(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "dispatch_profile.json"
    _write_iree_dispatch_profile(profile_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        iree_dispatch_profile_paths=[NamedPath("stage", profile_path)],
    )
    text = format_text_report(report)

    assert "IREE dispatch profiles:" in text
    assert "stage: kernels=2" in text
    assert "hot_kernel: count=2 total_ms=0.02 p50_ms=0.01" in text
    assert "cold_kernel: count=3 total_ms=0.006 p50_ms=0.002" in text


def test_text_report_contains_rocprof_kernel_trace_summary(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "kernel_trace.csv"
    _write_rocprof_kernel_trace(trace_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocprof_kernel_trace_paths=[NamedPath("profile", trace_path)],
    )
    text = format_text_report(report)

    assert "rocprof kernel traces:" in text
    assert "profile: dispatches=4 kernels=2" in text
    assert "hot_kernel: count=3 total_ms=0.0006 p50_ms=0.0002" in text
    assert "lds=2048 vgpr=64 sgpr=80 workgroup=128 grid=128" in text


def test_text_report_contains_rocprof_counter_collection_summary(
    tmp_path: Path,
) -> None:
    counter_path = tmp_path / "counter_collection.csv"
    _write_rocprof_counter_collection(counter_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocprof_counter_collection_paths=[NamedPath("profile", counter_path)],
    )
    text = format_text_report(report)

    assert "rocprof counter collections:" in text
    assert "profile: rows=6 dispatches=3 counters=2 kernels=2" in text
    assert "hot_kernel: count=2 total_ms=0.0003 p50_ms=0.0001" in text
    assert "SQ_INSTS_LDS_mean=20 SQ_INSTS_LDS_max=30" in text
    assert "LDSBankConflict_mean=3 LDSBankConflict_max=4" in text


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
    scorecard_by_metric = {entry["metric"]: entry for entry in comparison["scorecard"]}
    assert scorecard_by_metric["local_memory_bytes"]["category"] == "local_memory"
    assert scorecard_by_metric["local_memory_bytes"]["finding"] == "candidate_higher"
    assert scorecard_by_metric["local_memory_bytes"]["severity"] == 2
    assert scorecard_by_metric["vgpr_count"]["category"] == "resources"


def test_rocblas_trace_rows_participate_in_comparisons(tmp_path: Path) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("pytorch", rocblas_log_path)],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="pytorch/gemm_M13824_N4547_K4608_beta0.0",
                candidate="pytorch/gemm_M12288_N4547_K4608_beta0.0",
            )
        ],
    )

    comparison = comparisons[
        "pytorch/gemm_M13824_N4547_K4608_beta0.0="
        "pytorch/gemm_M12288_N4547_K4608_beta0.0"
    ]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["call_count"]["baseline"] == 68
    assert deltas_by_metric["call_count"]["candidate"] == 136
    assert deltas_by_metric["M"]["baseline"] == 13824
    assert deltas_by_metric["M"]["candidate"] == 12288


def test_rocblas_timing_rows_participate_in_comparisons(tmp_path: Path) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    _write_rocblas_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[
            NamedPath("baseline", rocblas_log_path),
            NamedPath("candidate", rocblas_log_path),
        ],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="baseline/gemm_M13824_N4547_K4608_beta0",
                candidate="candidate/gemm_M13824_N4547_K4608_beta0",
            )
        ],
    )

    comparison = comparisons[
        "baseline/gemm_M13824_N4547_K4608_beta0=candidate/gemm_M13824_N4547_K4608_beta0"
    ]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["operation_time_ns"]["baseline"] == 8503930.0
    assert deltas_by_metric["operation_time_ns"]["candidate"] == 8503930.0
    assert deltas_by_metric["M"]["baseline"] == 13824
    assert deltas_by_metric["N"]["baseline"] == 4547
    assert deltas_by_metric["K"]["baseline"] == 4608
    assert deltas_by_metric["hot_iters"]["candidate"] == 20


def test_rocblas_symbol_parameters_participate_in_comparisons(
    tmp_path: Path,
) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    _write_rocblas_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[
            NamedPath("baseline", rocblas_log_path),
            NamedPath("candidate", rocblas_log_path),
        ],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [ComparisonSpec(baseline="baseline", candidate="candidate")],
    )

    comparison = comparisons["baseline=candidate"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["macro_tile_x"]["baseline"] == 128
    assert deltas_by_metric["matrix_instruction_z"]["baseline"] == 16
    assert deltas_by_metric["flat_workgroup_size"]["candidate"] == 128


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
    assert "Comparison scorecard:" in text
    assert "baseline=loom: shared=2 baseline_metrics=2" in text
    assert "scorecard:" in text
    assert "local_memory_bytes [local_memory]: candidate_higher" in text
    assert "baseline=loom :: local_memory_bytes [local_memory]" in text
    assert "local_memory_bytes: baseline=1024 candidate=2048" in text
    assert "ratio=2x sources=amdhsa_metadata/compile_report" in text


def test_rocprof_metrics_participate_in_comparisons(tmp_path: Path) -> None:
    trace_path = tmp_path / "kernel_trace.csv"
    counter_path = tmp_path / "counter_collection.csv"
    _write_rocprof_kernel_trace(trace_path)
    _write_rocprof_counter_collection(counter_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocprof_kernel_trace_paths=[NamedPath("trace", trace_path)],
        rocprof_counter_collection_paths=[NamedPath("counters", counter_path)],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="trace/hot_kernel",
                candidate="counters/hot_kernel",
            )
        ],
    )

    comparison = comparisons["trace/hot_kernel=counters/hot_kernel"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["local_memory_bytes"]["baseline"] == 2048
    assert deltas_by_metric["local_memory_bytes"]["candidate"] == 2048
    assert deltas_by_metric["operation_time_ns"]["baseline"] == 200
    assert deltas_by_metric["operation_time_ns"]["candidate"] == 100
    assert "SQ_INSTS_LDS_mean" in comparison["missing_baseline_metrics"]


def test_compile_report_wait_and_source_metrics_participate_in_comparisons(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_payload = _compile_report_payload()
    candidate_payload = _compile_report_payload()
    candidate_entry = candidate_payload["entries"]["rows"][0]
    candidate_entry["wait_plan"]["action_count"] = 276
    candidate_entry["wait_reason_summary_rows"]["rows"][0]["summary"][
        "partial_wait_count"
    ] = 218
    candidate_payload["economics"]["memory"]["source_low"]["dynamic_packet_count"] = (
        156928
    )
    candidate_payload["economics"]["memory"]["source_low"]["dispatch_source"][
        "total_bytes"
    ] = 367172517888
    _write(baseline_path, json.dumps({"compile_report": baseline_payload}))
    _write(candidate_path, json.dumps({"compile_report": candidate_payload}))

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[
            NamedPath("baseline", baseline_path),
            NamedPath("candidate", candidate_path),
        ],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(baseline="baseline", candidate="candidate"),
            ComparisonSpec(
                baseline="baseline/wait/lds/amdgpu.read_result_reuse",
                candidate="candidate/wait/lds/amdgpu.read_result_reuse",
            ),
        ],
    )

    whole_comparison = comparisons["baseline=candidate"]
    whole_deltas = {delta["metric"]: delta for delta in whole_comparison["deltas"]}
    assert whole_deltas["wait_action_count"]["candidate"] == 276
    assert whole_deltas["source_low_dynamic_packet_count"]["ratio"] == 2
    assert whole_deltas["source_low_dispatch_source_total_bytes"]["ratio"] == 2
    whole_scorecard = {
        entry["metric"]: entry for entry in whole_comparison["scorecard"]
    }
    assert whole_scorecard["wait_action_count"]["category"] == "wait"
    assert (
        whole_scorecard["source_low_dynamic_packet_count"]["category"]
        == "source_memory"
    )

    reason_comparison = comparisons[
        "baseline/wait/lds/amdgpu.read_result_reuse="
        "candidate/wait/lds/amdgpu.read_result_reuse"
    ]
    reason_deltas = {delta["metric"]: delta for delta in reason_comparison["deltas"]}
    assert reason_deltas["partial_wait_count"]["ratio"] == 2


def test_weighted_symbol_metrics_participate_in_comparisons(
    tmp_path: Path,
) -> None:
    disassembly_path = tmp_path / "tensile.s"
    _write(
        disassembly_path,
        """
0000000000000000 <open_loop>:
      ds_read_b128 v[0:3], v0
      ds_write_b128 v0, v[0:3]
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("tensile", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        symbol_weight_specs=[SymbolWeightSpec("tensile", "open_loop", 5)],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="tensile/weighted_symbols",
                candidate="loom/dynamic",
            )
        ],
    )

    comparison = comparisons["tensile/weighted_symbols=loom/dynamic"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["local_memory_instruction_count"]["baseline"] == 10
    assert deltas_by_metric["local_memory_instruction_count"]["candidate"] == 99
    assert deltas_by_metric["local_memory_access_bytes"]["baseline"] == 160
    assert deltas_by_metric["wmma_count"]["baseline"] == 5
    assert comparison["scorecard"][0]["metric"] == "local_memory_instruction_count"
    assert comparison["scorecard"][0]["category"] == "local_memory"
    assert comparison["scorecard"][0]["severity"] == 9.9
    aggregate_scorecard = build_kernel_anatomy_comparison_scorecard(comparisons)
    assert aggregate_scorecard[0]["comparison"] == (
        "tensile/weighted_symbols=loom/dynamic"
    )
    assert aggregate_scorecard[0]["metric"] == "local_memory_instruction_count"


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


def test_main_emits_rocblas_replay_script(tmp_path: Path, capsys) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    assert (
        main(
            [
                "--rocblas-log",
                f"pytorch={rocblas_log_path}",
                "--rocblas-bench-executable",
                "/tools/rocblas-bench",
                "--rocblas-replay-arg=--cold_iters",
                "--rocblas-replay-arg",
                "9",
                "--format",
                "rocblas-replay",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "/tools/rocblas-bench --function gemm_ex" in text
    assert "--sizem 13824 --sizen 4547 --sizek 4608" in text
    assert "--atomics_not_allowed --cold_iters 9" in text


def test_main_emits_iree_dispatch_profile(tmp_path: Path, capsys) -> None:
    profile_path = tmp_path / "dispatch_profile.json"
    _write_iree_dispatch_profile(profile_path)

    assert (
        main(
            [
                "--iree-dispatch-profile",
                f"stage={profile_path}",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["iree_dispatch_profiles"]["stage"]["kernel_count"] == 2


def test_main_emits_rocprof_summaries(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "kernel_trace.csv"
    counter_path = tmp_path / "counter_collection.csv"
    _write_rocprof_kernel_trace(trace_path)
    _write_rocprof_counter_collection(counter_path)

    assert (
        main(
            [
                "--rocprof-kernel-trace",
                f"profile={trace_path}",
                "--rocprof-counter-collection",
                f"profile={counter_path}",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["rocprof_kernel_traces"]["profile"]["dispatch_count"] == 4
    assert report["rocprof_counter_collections"]["profile"]["counter_count"] == 2


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
    assert report["comparison_scorecard"][0]["comparison"] == "baseline=loom"


def test_main_emits_weighted_symbols(tmp_path: Path, capsys) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <main_loop>:
      ds_read_b128 v[0:3], v0
""",
    )

    assert (
        main(
            [
                "--disassembly",
                f"asm={disassembly_path}",
                "--symbol-weight",
                "asm=main_loop=9",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    weighted_symbols = report["disassemblies"]["asm"]["weighted_symbols"]
    assert weighted_symbols["summary"]["family_counts"]["ds_read"] == 9
