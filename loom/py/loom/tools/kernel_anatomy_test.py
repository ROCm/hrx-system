# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import struct
from pathlib import Path

from loom.tools.kernel_anatomy import (
    ComparisonSpec,
    NamedPath,
    SymbolWeightSpec,
    WeightedSymbolGroupSpec,
    attach_kernel_anatomy_benchmark_target_listings,
    build_kernel_anatomy_best_candidate_rows,
    build_kernel_anatomy_comparison_scorecard,
    build_kernel_anatomy_comparison_verdicts,
    build_kernel_anatomy_comparisons,
    build_kernel_anatomy_duplicate_candidate_rows,
    build_kernel_anatomy_optimization_frontier,
    build_kernel_anatomy_report,
    build_kernel_anatomy_rocblas_benchmark_shape_comparison_specs,
    build_kernel_anatomy_structural_bottleneck_rows,
    format_rocblas_replay_script,
    format_rocblas_solution_trace_script,
    format_summary_report,
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


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _pack_msgpack(value) -> bytes:
    if value is None:
        return b"\xc0"
    if value is False:
        return b"\xc2"
    if value is True:
        return b"\xc3"
    if isinstance(value, int):
        if 0 <= value <= 0x7F:
            return bytes([value])
        if -32 <= value < 0:
            return bytes([value + 0x100])
        if 0 <= value <= 0xFF:
            return b"\xcc" + struct.pack(">B", value)
        if 0 <= value <= 0xFFFF:
            return b"\xcd" + struct.pack(">H", value)
        if 0 <= value <= 0xFFFFFFFF:
            return b"\xce" + struct.pack(">I", value)
        if -(1 << 31) <= value < 0:
            return b"\xd2" + struct.pack(">i", value)
        return b"\xd3" + struct.pack(">q", value)
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    if isinstance(value, str):
        data = value.encode("utf-8")
        if len(data) <= 31:
            return bytes([0xA0 | len(data)]) + data
        if len(data) <= 0xFF:
            return b"\xd9" + struct.pack(">B", len(data)) + data
        if len(data) <= 0xFFFF:
            return b"\xda" + struct.pack(">H", len(data)) + data
        return b"\xdb" + struct.pack(">I", len(data)) + data
    if isinstance(value, list):
        payload = b"".join(_pack_msgpack(item) for item in value)
        if len(value) <= 15:
            return bytes([0x90 | len(value)]) + payload
        if len(value) <= 0xFFFF:
            return b"\xdc" + struct.pack(">H", len(value)) + payload
        return b"\xdd" + struct.pack(">I", len(value)) + payload
    if isinstance(value, dict):
        payload = b"".join(
            _pack_msgpack(key) + _pack_msgpack(item) for key, item in value.items()
        )
        if len(value) <= 15:
            return bytes([0x80 | len(value)]) + payload
        if len(value) <= 0xFFFF:
            return b"\xde" + struct.pack(">H", len(value)) + payload
        return b"\xdf" + struct.pack(">I", len(value)) + payload
    raise TypeError(type(value).__name__)


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
                        "private_memory_count": 7,
                        "private_read_byte_count": 96,
                        "private_write_byte_count": 64,
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
                            "x": 128,
                            "y": 1,
                            "z": 1,
                            "flat": 128,
                        },
                        "workgroup_count": {
                            "x": 36,
                            "y": 2,
                            "z": 1,
                            "flat": 72,
                        },
                        "dispatch_workitem_count": 9216,
                    },
                },
            ],
        },
    }


def _linear_compile_report_payload(
    *, output_size: int, token_count: int, input_size: int
) -> dict:
    payload = _compile_report_payload()
    payload["config_bindings"] = {
        "count": 4,
        "rows": [
            {
                "index": 0,
                "key": "benchmark.linear_wmma.token_count",
                "value": str(token_count),
            },
            {
                "index": 1,
                "key": "benchmark.linear_wmma.dispatch_token_count",
                "value": str(token_count),
            },
            {
                "index": 2,
                "key": "benchmark.linear_wmma.input_size",
                "value": str(input_size),
            },
            {
                "index": 3,
                "key": "benchmark.linear_wmma.output_size",
                "value": str(output_size),
            },
        ],
    }
    return payload


def _multi_entry_compile_report_payload() -> dict:
    report = _compile_report_payload()
    first_entry = dict(report["entries"]["rows"][0])
    first_entry["function"] = "wrong_kernel"
    first_entry["instruction_count"] = 999
    second_entry = dict(report["entries"]["rows"][0])
    second_entry["function"] = "selected_kernel"
    second_entry["instruction_count"] = 321
    report["entries"]["rows"] = [first_entry, second_entry]
    return report


def _write_benchmark_jsonl(path: Path) -> None:
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "device",
                        "run_id": "r0",
                        "device_uri": "amdgpu",
                        "driver": "amdgpu",
                        "provider": "amdgpu-hal",
                        "target_family": "AMDGPU",
                        "device_spec": {
                            "physical_devices": [
                                {
                                    "display_name": "gfx1100",
                                    "backend_path": "gfx1100",
                                },
                            ],
                            "dispatch": {
                                "execution": {
                                    "maximum_workgroup_local_memory_size": 4096,
                                    "maximum_workgroup_local_memory_size_optin": 4096,
                                },
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "candidate_index": 0,
                        "benchmark": "bench_kernel",
                        "case": "case_kernel",
                        "entry": "loom_kernel",
                        "state": "ok",
                        "diagnostic_error_count": 0,
                        "diagnostic_warning_count": 2,
                        "diagnostic_remark_count": 0,
                        "compile_report": _compile_report_payload(),
                    }
                ),
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
                json.dumps(
                    {
                        "row": "benchmark.repetition",
                        "candidate_id": "c1",
                        "candidate_index": 1,
                        "comparison_group": "bench_kernel",
                        "baseline_candidate_id": "c0",
                        "method": "ABABA",
                        "order_index": 1,
                        "repetition_index": 0,
                        "schedule_token": "B",
                        "benchmark_result": {
                            "benchmark": "bench_kernel_candidate",
                            "case": "case_kernel_candidate",
                            "state": "ok",
                            "sample_compilation": "once",
                            "correctness": {
                                "sample_count": 1,
                                "failed_sample_count": 0,
                            },
                            "measurement": {
                                "operation_timing_ns": {
                                    "count": 5,
                                    "p50": 1111000,
                                    "p90": 1125000,
                                },
                                "timing_interpretation": {
                                    "warnings": [],
                                },
                            },
                            "compile_report": _compile_report_payload(),
                        },
                    }
                ),
                json.dumps(
                    {
                        "row": "comparison",
                        "run_id": "r0",
                        "comparison_group": "bench_kernel",
                        "method": "ABABA",
                        "baseline_candidate_id": "c0",
                        "candidate_id": "c1",
                        "baseline_repetition_count": 4,
                        "candidate_repetition_count": 3,
                        "baseline_p50_ns": 1234000,
                        "candidate_p50_ns": 1111000,
                        "baseline_p90_ns": 1250000,
                        "candidate_p90_ns": 1125000,
                        "baseline_p50_spread_ppm": 5000,
                        "candidate_p50_spread_ppm": 6000,
                        "baseline_p90_spread_ppm": 7000,
                        "candidate_p90_spread_ppm": 8000,
                        "ratio_p50": 0.900324,
                        "speedup_p50": 1.110711,
                        "ratio_p90": 0.9,
                        "speedup_p90": 1.111111,
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_compare_only_benchmark_jsonl(path: Path) -> None:
    baseline_report = _compile_report_payload()
    candidate_report = json.loads(json.dumps(_compile_report_payload()))
    candidate_entry = candidate_report["entries"]["rows"][0]
    candidate_entry["instruction_count"] = 100
    candidate_entry["local_memory_bytes"] = 4096
    candidate_entry["dynamic_instruction_mix"]["local_memory_count"] = 198
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "candidate_index": 0,
                        "benchmark": "bench_kernel",
                        "case": "case_kernel",
                        "entry": "loom_kernel",
                        "state": "ok",
                        "diagnostic_error_count": 0,
                        "diagnostic_warning_count": 0,
                        "diagnostic_remark_count": 0,
                        "compile_report": baseline_report,
                    }
                ),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c1",
                        "candidate_index": 1,
                        "benchmark": "bench_kernel_candidate",
                        "case": "case_kernel",
                        "entry": "loom_kernel",
                        "state": "ok",
                        "diagnostic_error_count": 0,
                        "diagnostic_warning_count": 0,
                        "diagnostic_remark_count": 0,
                        "compile_report": candidate_report,
                    }
                ),
                json.dumps(
                    {
                        "row": "comparison",
                        "run_id": "r0",
                        "comparison_group": "bench_kernel",
                        "method": "ABABA",
                        "baseline_candidate_id": "c0",
                        "candidate_id": "c1",
                        "baseline_repetition_count": 4,
                        "candidate_repetition_count": 3,
                        "baseline_p50_ns": 1_000_000,
                        "candidate_p50_ns": 1_100_000,
                        "baseline_p90_ns": 1_100_000,
                        "candidate_p90_ns": 1_210_000,
                        "baseline_p50_spread_ppm": 5000,
                        "candidate_p50_spread_ppm": 6000,
                        "baseline_p90_spread_ppm": 7000,
                        "candidate_p90_spread_ppm": 8000,
                        "ratio_p50": 1.1,
                        "speedup_p50": 0.909091,
                        "ratio_p90": 1.1,
                        "speedup_p90": 0.909091,
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_duplicate_candidate_benchmark_jsonl(path: Path) -> None:
    compile_report = _compile_report_payload()
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "candidate_index": 0,
                        "benchmark": "bench_kernel",
                        "case": "case_kernel",
                        "entry": "baseline_entry",
                        "state": "ok",
                        "diagnostic_error_count": 0,
                        "diagnostic_warning_count": 0,
                        "diagnostic_remark_count": 0,
                        "compile_report": compile_report,
                    }
                ),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c1",
                        "candidate_index": 1,
                        "benchmark": "bench_kernel_candidate",
                        "case": "case_kernel",
                        "entry": "candidate_entry",
                        "state": "ok",
                        "diagnostic_error_count": 0,
                        "diagnostic_warning_count": 0,
                        "diagnostic_remark_count": 0,
                        "compile_report": compile_report,
                    }
                ),
                json.dumps(
                    {
                        "row": "comparison",
                        "run_id": "r0",
                        "comparison_group": "bench_kernel",
                        "method": "ABABA",
                        "baseline_candidate_id": "c0",
                        "candidate_id": "c1",
                        "baseline_repetition_count": 4,
                        "candidate_repetition_count": 3,
                        "baseline_p50_ns": 1_000_000,
                        "candidate_p50_ns": 1_007_000,
                        "baseline_p90_ns": 1_100_000,
                        "candidate_p90_ns": 1_107_000,
                        "baseline_p50_spread_ppm": 5000,
                        "candidate_p50_spread_ppm": 6000,
                        "baseline_p90_spread_ppm": 7000,
                        "candidate_p90_spread_ppm": 8000,
                        "ratio_p50": 1.007,
                        "speedup_p50": 0.993049,
                        "ratio_p90": 1.006364,
                        "speedup_p90": 0.993677,
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_skipped_benchmark_compile_jsonl(path: Path) -> None:
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "candidate_index": 0,
                        "benchmark": "skipped_kernel",
                        "case": "skipped_case",
                        "entry": "loom_kernel",
                        "state": "ok",
                        "diagnostic_error_count": 0,
                        "diagnostic_warning_count": 41,
                        "diagnostic_remark_count": 0,
                        "compile_report": _compile_report_payload(),
                    }
                ),
                json.dumps(
                    {
                        "row": "benchmark",
                        "candidate_id": "c0",
                        "candidate_index": 0,
                        "benchmark_result": {
                            "benchmark": "skipped_kernel",
                            "case": "skipped_case",
                            "state": "skipped",
                            "sample_compilation": "once",
                            "correctness": {
                                "sample_count": 1,
                                "failed_sample_count": 1,
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_single_benchmark_jsonl(
    path: Path, benchmark_name: str, compile_report: dict, p50_ns: int
) -> None:
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "benchmark",
                        "candidate_id": "c0",
                        "benchmark_result": {
                            "benchmark": benchmark_name,
                            "case": f"{benchmark_name}_case",
                            "state": "ok",
                            "correctness": {
                                "sample_count": 1,
                                "failed_sample_count": 0,
                            },
                            "measurement": {
                                "operation_timing_ns": {
                                    "count": 5,
                                    "p50": p50_ns,
                                },
                            },
                            "compile_report": compile_report,
                        },
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_resource_violation_benchmark_jsonl(path: Path) -> None:
    compile_report = _compile_report_payload()
    compile_report["entries"]["rows"][0]["local_memory_bytes"] = 8192
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "device",
                        "run_id": "r0",
                        "device_uri": "amdgpu",
                        "device_spec": {
                            "physical_devices": [
                                {
                                    "display_name": "gfx1100",
                                },
                            ],
                            "dispatch": {
                                "execution": {
                                    "maximum_workgroup_local_memory_size": 4096,
                                },
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "benchmark": "oversized_kernel",
                        "case": "oversized_case",
                        "entry": "oversized_entry",
                        "state": "ok",
                        "compile_report": compile_report,
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_multi_entry_benchmark_jsonl(path: Path) -> None:
    compile_report = _multi_entry_compile_report_payload()
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "plan",
                        "candidate_id": "c0",
                        "benchmark": "selected_benchmark",
                        "case": "selected_case",
                        "actual_entry": "selected_kernel",
                    }
                ),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "benchmark": "selected_benchmark",
                        "case": "selected_case",
                        "entry": "selected_kernel",
                        "state": "ok",
                        "compile_report": compile_report,
                    }
                ),
                json.dumps(
                    {
                        "row": "benchmark",
                        "candidate_id": "c0",
                        "benchmark_result": {
                            "benchmark": "selected_benchmark",
                            "case": "selected_case",
                            "state": "ok",
                            "correctness": {
                                "sample_count": 1,
                                "failed_sample_count": 0,
                            },
                            "measurement": {
                                "operation_timing_ns": {
                                    "p50": 1234000,
                                    "p90": 1250000,
                                },
                            },
                            "compile_report": compile_report,
                        },
                    }
                ),
            ]
        )
        + "\n",
    )


def _write_target_listing_benchmark_jsonl(path: Path, listing_path: Path) -> None:
    compile_report = _multi_entry_compile_report_payload()
    _write(
        path,
        "\n".join(
            [
                json.dumps({"row": "run", "run_id": "r0"}),
                json.dumps(
                    {
                        "row": "compile",
                        "run_id": "r0",
                        "candidate_id": "c0",
                        "benchmark": "selected_benchmark",
                        "case": "selected_case",
                        "entry": "selected_kernel",
                        "state": "ok",
                        "target_listing_path": listing_path.as_posix(),
                        "compile_report": compile_report,
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


def _write_tensile_catalog(path: Path) -> None:
    _write_bytes(
        path,
        _pack_msgpack(
            {
                "solutions": [
                    {
                        "name": _TENSILE_SYMBOL,
                        "index": 1140853047,
                        "libraryLogicIndex": 44,
                        "problemType": {
                            "operationIdentifier": (
                                "Contraction_l_Alik_Bljk_Cijk_Dijk"
                            ),
                            "aType": "BFloat16",
                            "bType": "BFloat16",
                            "cType": "BFloat16",
                            "dType": "BFloat16",
                            "useBeta": False,
                            "highPrecisionAccumulate": True,
                            "stridedBatched": True,
                            "f32XdlMathOp": "Float",
                        },
                        "sizeMapping": {
                            "workGroup": [32, 4, 1],
                            "macroTile": [128, 128, 1],
                            "threadTile": [4, 64],
                            "depthU": 16,
                            "globalSplitU": 1,
                            "workGroupMapping": 4,
                        },
                    },
                    {
                        "name": "tiny_kernel",
                        "index": 123,
                        "libraryLogicIndex": 0,
                        "problemType": {},
                        "sizeMapping": {
                            "workGroup": [32, 4, 1],
                            "macroTile": [32, 32, 1],
                            "threadTile": [1, 16],
                            "depthU": 16,
                        },
                    },
                ],
                "library": {
                    "type": "Problem",
                    "rows": [
                        {
                            "predicate": {"type": "TruePred"},
                            "library": {
                                "type": "Matching",
                                "properties": [
                                    {"type": "FreeSizeA", "index": 0},
                                    {"type": "FreeSizeB", "index": 0},
                                    {"type": "BoundSize", "index": 0},
                                ],
                                "table": [
                                    {
                                        "key": [6144, 4096, 4096],
                                        "index": 1140853047,
                                        "speed": 1.0,
                                    }
                                ],
                            },
                        }
                    ],
                },
            }
        ),
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
    matrix_symbol = report["disassemblies"]["asm"]["matrix_symbols"][0]
    assert matrix_symbol["symbol"] == "big"
    assert matrix_symbol["matrix_instruction_count"] == 2
    assert matrix_symbol["instruction_count"] == 3
    assert matrix_symbol["instructions_per_matrix_instruction"] == 1.5
    ordered_symbols = report["disassemblies"]["asm"]["ordered_symbols"]
    assert [symbol["symbol"] for symbol in ordered_symbols] == ["small", "big"]
    assert [symbol["address"] for symbol in ordered_symbols] == [0, 0x100]
    summary = format_summary_report(report)
    assert "Matrix-heavy disassembly blocks:" in summary
    assert "asm/big: instructions=3 matrix=2" in summary


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
    assert benchmark_report["comparison_count"] == 1
    assert benchmark_report["compile_count"] == 1
    assert benchmark_report["device_count"] == 1
    assert benchmark_report["repetition_count"] == 1
    assert not benchmark_report["resource_findings"]
    assert benchmark_report["devices"][0]["maximum_workgroup_local_memory_size"] == 4096
    assert benchmark_report["compiles"][0]["diagnostic_warning_count"] == 2
    benchmark = benchmark_report["benchmarks"][0]
    assert benchmark["benchmark"] == "bench_kernel"
    assert benchmark["timing_ns"]["p50"] == 1234000
    assert benchmark["correctness"]["failed_sample_count"] == 0
    assert benchmark["compile_report"]["instruction_count"] == 123
    assert (
        benchmark["compile_report"]["dynamic_instruction_mix"]["local_memory_count"]
        == 99
    )
    repetition = benchmark_report["repetitions"][0]
    assert repetition["comparison_group"] == "bench_kernel"
    assert repetition["schedule_token"] == "B"
    assert repetition["timing_ns"]["p50"] == 1111000
    comparison = benchmark_report["comparisons"][0]
    assert comparison["method"] == "ABABA"
    assert comparison["baseline_repetition_count"] == 4
    assert comparison["candidate_repetition_count"] == 3
    assert comparison["ratio_p50"] == 0.900324

    text_report = format_text_report(report)
    assert "repetitions=1 comparisons=1" in text_report
    assert "benchmark comparisons:" in text_report
    assert "ratio_p50=0.900324x" in text_report
    assert "benchmark repetitions:" in text_report
    assert "B0: bench_kernel_candidate" in text_report


def test_benchmark_embedded_compile_reports_participate_in_comparisons(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_payload = _compile_report_payload()
    candidate_payload = _compile_report_payload()
    candidate_entry = candidate_payload["entries"]["rows"][0]
    candidate_entry["instruction_count"] = 100
    candidate_entry["local_memory_bytes"] = 4096
    candidate_entry["dynamic_instruction_mix"]["local_memory_count"] = 198
    candidate_payload["economics"]["memory"]["source_low"]["dynamic_packet_count"] = (
        156928
    )
    _write_single_benchmark_jsonl(
        baseline_path, "baseline_kernel", baseline_payload, 1_000_000
    )
    _write_single_benchmark_jsonl(
        candidate_path, "candidate_kernel", candidate_payload, 4_000_000
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[
            NamedPath("baseline", baseline_path),
            NamedPath("candidate", candidate_path),
        ],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report, [ComparisonSpec(baseline="baseline", candidate="candidate")]
    )

    comparison = comparisons["baseline=candidate"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["operation_time_ns"]["ratio"] == 4
    assert deltas_by_metric["local_memory_bytes"]["ratio"] == 2
    assert deltas_by_metric["dynamic_local_memory_count"]["ratio"] == 2
    assert deltas_by_metric["source_low_dynamic_packet_count"]["ratio"] == 2
    assert (
        deltas_by_metric["local_memory_bytes"]["candidate_source"]
        == "benchmark_compile_report"
    )
    findings_by_kind_role = {
        (finding["kind"], finding["role"], finding["metric"]): finding
        for finding in comparison["gap_findings"]
    }
    assert (
        "time_gap",
        "candidate_slower",
        "operation_time_ns",
    ) in findings_by_kind_role
    assert (
        "candidate_cost",
        "partial_time_gap",
        "dynamic_local_memory_count",
    ) in findings_by_kind_role
    assert (
        "candidate_advantage",
        "counter_improved",
        "instruction_count",
    ) in findings_by_kind_role


def test_compare_only_benchmark_rows_participate_in_comparisons(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "compare_only.jsonl"
    _write_compare_only_benchmark_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="bench/c0/bench_kernel",
                candidate="bench/c1/bench_kernel",
            )
        ],
    )

    comparison = comparisons["bench/c0/bench_kernel=bench/c1/bench_kernel"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["operation_time_ns"]["ratio"] == 1.1
    assert (
        deltas_by_metric["operation_time_ns"]["baseline_source"]
        == "benchmark_comparison"
    )
    assert (
        deltas_by_metric["operation_time_ns"]["candidate_source"]
        == "benchmark_comparison"
    )
    assert deltas_by_metric["local_memory_bytes"]["ratio"] == 2
    assert (
        deltas_by_metric["local_memory_bytes"]["candidate_source"]
        == "benchmark_compile_report"
    )
    assert deltas_by_metric["dynamic_local_memory_count"]["ratio"] == 2
    assert deltas_by_metric["instruction_count"]["candidate"] == 100
    scorecard_metrics = {entry["metric"] for entry in comparison["scorecard"]}
    assert "p90_spread_ppm" not in scorecard_metrics
    assert "repetition_count" not in scorecard_metrics

    verdicts = build_kernel_anatomy_comparison_verdicts(comparisons)
    assert verdicts[0]["status"] == "slower_with_structural_cost"
    assert verdicts[0]["primary_cost"]["category"] == "local_memory"

    frontier = build_kernel_anatomy_optimization_frontier(
        report,
        ["bench/c0/bench_kernel"],
        ["/c1/"],
    )
    assert [entry["candidate"] for entry in frontier] == ["bench/c1/bench_kernel"]
    assert frontier[0]["status"] == "slower"


def test_structural_bottleneck_rows_group_recurring_primary_costs(
    tmp_path: Path,
) -> None:
    baseline_report = _compile_report_payload()
    candidate_report = json.loads(json.dumps(_compile_report_payload()))
    candidate_report["entries"]["rows"][0]["instruction_count"] = 100
    candidate_report["entries"]["rows"][0]["dynamic_instruction_mix"][
        "local_memory_count"
    ] = 198
    _write_single_benchmark_jsonl(
        tmp_path / "baseline_a.jsonl",
        "baseline_a",
        baseline_report,
        1_000_000,
    )
    _write_single_benchmark_jsonl(
        tmp_path / "candidate_a.jsonl",
        "candidate_a",
        candidate_report,
        3_000_000,
    )
    _write_single_benchmark_jsonl(
        tmp_path / "baseline_b.jsonl",
        "baseline_b",
        baseline_report,
        2_000_000,
    )
    _write_single_benchmark_jsonl(
        tmp_path / "candidate_b.jsonl",
        "candidate_b",
        candidate_report,
        8_000_000,
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[
            NamedPath("baseline_a", tmp_path / "baseline_a.jsonl"),
            NamedPath("candidate_a", tmp_path / "candidate_a.jsonl"),
            NamedPath("baseline_b", tmp_path / "baseline_b.jsonl"),
            NamedPath("candidate_b", tmp_path / "candidate_b.jsonl"),
        ],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(baseline="baseline_a", candidate="candidate_a"),
            ComparisonSpec(baseline="baseline_b", candidate="candidate_b"),
        ],
    )

    rows = build_kernel_anatomy_structural_bottleneck_rows(comparisons)

    assert rows[0]["metric"] == "dynamic_local_memory_count"
    assert rows[0]["category"] == "local_memory"
    assert rows[0]["comparison_count"] == 2
    assert rows[0]["status_counts"] == {"slower_with_structural_cost": 2}
    assert rows[0]["cost_ratio_geomean"] == 2
    assert rows[0]["time_ratio_geomean"] > 3
    assert len(rows[0]["comparisons"]) == 2

    report["comparisons"] = comparisons
    summary = format_summary_report(report)
    assert "Recurring structural bottlenecks:" in summary
    assert "dynamic_local_memory_count[local_memory]: comparisons=2" in summary


def test_duplicate_candidate_rows_find_same_compile_signature(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "duplicate_candidates.jsonl"
    _write_duplicate_candidate_benchmark_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )

    rows = build_kernel_anatomy_duplicate_candidate_rows(report)

    assert len(rows) == 1
    assert rows[0]["report"] == "bench"
    assert rows[0]["comparison_group"] == "bench_kernel"
    assert rows[0]["baseline_candidate_id"] == "c0"
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["baseline_entry"] == "baseline_entry"
    assert rows[0]["candidate_entry"] == "candidate_entry"
    assert rows[0]["ratio_p50"] == 1.007
    assert rows[0]["signature_digest"]


def test_build_kernel_anatomy_report_flags_benchmark_resource_violations(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "results.jsonl"
    _write_resource_violation_benchmark_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )

    benchmark_report = report["loom_benchmarks"]["bench"]
    finding = benchmark_report["resource_findings"][0]
    assert finding["severity"] == "error"
    assert finding["resource"] == "workgroup_local_memory"
    assert finding["required_bytes"] == 8192
    assert finding["limit_bytes"] == 4096
    assert finding["overage_bytes"] == 4096
    assert finding["device"] == "gfx1100"
    assert finding["benchmark"] == "oversized_kernel"
    assert finding["entry"] == "oversized_entry"

    text_report = format_text_report(report)
    assert "resource error: workgroup_local_memory" in text_report
    assert "required=8192 limit=4096 overage=4096" in text_report


def test_build_kernel_anatomy_report_selects_benchmark_compile_entry(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "results.jsonl"
    _write_multi_entry_benchmark_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )

    benchmark_report = report["loom_benchmarks"]["bench"]
    benchmark = benchmark_report["benchmarks"][0]
    compile_row = benchmark_report["compiles"][0]
    assert benchmark["compile_report"]["function"] == "selected_kernel"
    assert benchmark["compile_report"]["instruction_count"] == 321
    assert compile_row["compile_report"]["function"] == "selected_kernel"
    assert compile_row["compile_report"]["instruction_count"] == 321

    text_report = format_text_report(report)
    assert "benchmark compiles:" in text_report
    assert "entry=selected_kernel" in text_report
    assert "instructions=321" in text_report


def test_benchmark_target_listings_attach_selected_entry_disassembly(
    tmp_path: Path,
) -> None:
    listing_path = tmp_path / "target_listing.amdgpu-assembly"
    _write(
        listing_path,
        """
0000000000000000 <wrong_kernel>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000100 <selected_kernel>:
      global_load_b128 v[0:3], v0
      ds_read_b128 v[4:7], v8
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )
    benchmark_path = tmp_path / "results.jsonl"
    _write_target_listing_benchmark_jsonl(benchmark_path, listing_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )
    attach_kernel_anatomy_benchmark_target_listings(report, top_symbol_count=4)

    disassembly = report["disassemblies"]["bench/c0/selected_benchmark"]
    assert disassembly["symbol_count"] == 1
    assert disassembly["top_symbols"][0]["symbol"] == "selected_kernel"
    assert disassembly["matrix_symbols"][0]["matrix_instruction_count"] == 1
    assert disassembly["matrix_symbols"][0]["local_memory_instruction_count"] == 1
    assert disassembly["matrix_symbols"][0]["top_local_memory_mnemonics"] == [
        {"mnemonic": "ds_read_b128", "count": 1}
    ]
    assert disassembly["matrix_symbols"][0]["top_device_memory_mnemonics"] == [
        {"mnemonic": "global_load_b128", "count": 1}
    ]

    summary = format_summary_report(report)
    assert "Matrix-heavy disassembly blocks:" in summary
    assert "bench/c0/selected_benchmark/selected_kernel" in summary
    assert "instructions=3 matrix=1 local=1 device=1" in summary
    assert "local_ops=ds_read_b128:1 device_ops=global_load_b128:1" in summary


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


def test_build_kernel_anatomy_report_extracts_tensile_catalog(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "TensileLibrary.dat"
    rocblas_log_path = tmp_path / "rocblas.log"
    compile_report_path = tmp_path / "compile_report.json"
    _write_tensile_catalog(catalog_path)
    _write_rocblas_log(rocblas_log_path)
    compile_payload = _compile_report_payload()
    compile_payload["entries"]["rows"][0]["workload"]["workgroup_size"] = {
        "x": 32,
        "y": 4,
        "z": 1,
        "flat": 128,
    }
    _write(
        compile_report_path,
        json.dumps({"compile_report": compile_payload}),
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        rocblas_log_paths=[NamedPath("tensile", rocblas_log_path)],
        tensile_catalog_paths=[NamedPath("catalog", catalog_path)],
    )

    catalog = report["tensile_catalogs"]["catalog"]
    assert catalog["solution_count"] == 2
    assert catalog["library"]["type"] == "Problem"
    assert catalog["library"]["rows"][0]["library_type"] == "Matching"
    assert catalog["library"]["rows"][0]["table_row_count"] == 1
    solution = catalog["solutions"][1]
    assert solution["library_logic_index"] == 44
    assert solution["index"] == 1140853047
    assert solution["symbol_parameters"]["macro_tile"] == {
        "x": 128,
        "y": 128,
        "z": 16,
    }
    assert solution["size_mapping"]["workGroup"] == [32, 4, 1]
    shape_match = catalog["rocblas_shape_matches"][0]
    assert shape_match["shape_key"] == "M13824_N4547_K4608_beta0"
    assert shape_match["problem_key"] == [13824, 4547, 4608]
    assert shape_match["matches"][0]["library_logic_index"] == 44
    assert shape_match["matches"][0]["key"] == [6144, 4096, 4096]

    comparisons = build_kernel_anatomy_comparisons(
        report,
        [ComparisonSpec(baseline="catalog/solution44", candidate="tensile")],
    )
    comparison = comparisons["catalog/solution44=tensile"]
    assert comparison["shared_metric_count"] >= 10
    assert not comparison["scorecard"]

    catalog_to_loom = build_kernel_anatomy_comparisons(
        report,
        [ComparisonSpec(baseline="catalog/solution44", candidate="loom")],
    )["catalog/solution44=loom"]
    loom_deltas = {delta["metric"]: delta for delta in catalog_to_loom["deltas"]}
    assert loom_deltas["flat_workgroup_size"]["baseline"] == 128
    assert loom_deltas["flat_workgroup_size"]["candidate"] == 128
    assert "tensile_workgroup_size_y" in catalog_to_loom["missing_candidate_metrics"]
    match_to_loom = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="catalog/match/M13824_N4547_K4608_beta0",
                candidate="loom",
            )
        ],
    )["catalog/match/M13824_N4547_K4608_beta0=loom"]
    match_deltas = {delta["metric"]: delta for delta in match_to_loom["deltas"]}
    assert match_deltas["flat_workgroup_size"]["baseline"] == 128
    assert match_deltas["flat_workgroup_size"]["candidate"] == 128

    text = format_text_report(report)
    assert "Tensile catalogs:" in text
    assert "catalog: solutions=2 library=Problem rows=1" in text
    assert "Matching: properties=[FreeSizeA[0], FreeSizeB[0], BoundSize[0]]" in text
    assert "rocBLAS shape matches:" in text
    assert "catalog/timing_row M13824_N4547_K4608_beta0" not in text
    assert "tensile M13824_N4547_K4608_beta0" in text
    assert "rank 0: solution=44 index=1140853047" in text
    assert "solution 44: index=1140853047" in text
    assert "MT=128x128x16" in text


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


def test_weighted_symbol_groups_estimate_named_phases(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <prologue>:
      buffer_load_b128 v[0:3], v0, s[0:3], 0 offen
0000000000000100 <open_loop_even>:
      buffer_load_b128 v[0:3], v0, s[0:3], 0 offen
      ds_write_b128 v0, v[0:3]
      ds_read_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000200 <open_loop_odd>:
      buffer_load_b128 v[0:3], v0, s[0:3], 0 offen
      ds_write_b128 v0, v[0:3]
      ds_read_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000300 <writeback>:
      buffer_store_b128 v0, v[0:3], s[0:3], 0 offen
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("tensile", disassembly_path)],
        compile_report_paths=[],
        weighted_symbol_group_specs=[
            WeightedSymbolGroupSpec("tensile", "open_loop", "open_loop_even", 4),
            WeightedSymbolGroupSpec("tensile", "open_loop", "open_loop_odd", 3),
            WeightedSymbolGroupSpec("tensile", "writeback", "writeback", 1),
        ],
    )

    groups = report["disassemblies"]["tensile"]["weighted_symbol_groups"]
    assert set(groups) == {"open_loop", "writeback"}
    open_loop = groups["open_loop"]
    assert open_loop["rule_count"] == 2
    assert open_loop["matched_symbol_count"] == 2
    assert open_loop["summary"]["instruction_count"] == 28
    assert open_loop["summary"]["family_counts"]["v_wmma"] == 7
    assert open_loop["summary"]["family_counts"]["ds_read"] == 7
    assert open_loop["summary"]["family_counts"]["buffer_load"] == 7
    writeback = groups["writeback"]
    assert writeback["summary"]["instruction_count"] == 1
    assert writeback["summary"]["family_counts"]["buffer_store"] == 1


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


def test_text_report_contains_weighted_symbol_groups(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <open_loop>:
      ds_read_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000100 <writeback>:
      buffer_store_b128 v0, v[0:3], s[0:3], 0 offen
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[],
        weighted_symbol_group_specs=[
            WeightedSymbolGroupSpec("asm", "open_loop", "open_loop", 4),
            WeightedSymbolGroupSpec("asm", "writeback", "writeback", 1),
        ],
    )
    text = format_text_report(report)

    assert "weighted symbol groups:" in text
    assert "weighted symbols: rules=1 group=open_loop matches=1 instructions=8" in text
    assert "wmma=4" in text
    assert "weighted symbols: rules=1 group=writeback matches=1 instructions=1" in text
    assert "buffer_store=1" in text


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
    assert "failed_samples=0/1 instructions=123 code_bytes=456" in text
    assert "local_bytes=2048 vgpr=64 occupancy=50%" in text
    assert "wmma=2 valu=17 dynamic_local=99 dynamic_private=7 warnings=2" in text


def test_text_report_uses_compile_row_for_skipped_benchmark(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "results.jsonl"
    _write_skipped_benchmark_compile_jsonl(benchmark_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )
    text = format_text_report(report)

    assert "skipped_kernel: state=skipped p50_ms=?" in text
    assert "failed_samples=1/1 instructions=123 code_bytes=456" in text
    assert "dynamic_local=99 dynamic_private=7 warnings=41" in text


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


def test_rocblas_solution_trace_script_contains_capture_environment(
    tmp_path: Path,
) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        rocblas_log_paths=[NamedPath("pytorch", rocblas_log_path)],
    )
    text = format_rocblas_solution_trace_script(
        report,
        executable="/opt/rocm/bin/rocblas-bench",
        extra_arguments=["--iters", "3"],
    )

    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "# pytorch M12288_N4547_K4608_beta0.0 calls=136" in text
    assert "ROCBLAS_LAYER=0 TENSILE_SOLUTION_SELECTION_TRACE=1 TENSILE_DB=1" in text
    assert "/opt/rocm/bin/rocblas-bench --function gemm_ex" in text
    assert "--sizem 12288 --sizen 4547 --sizek 4608" in text
    assert "--solution_index 0" not in text
    assert "--cold_iters 0 --iters 3" in text
    assert "--iters 1" not in text

    pinned_text = format_rocblas_solution_trace_script(
        report,
        executable="/opt/rocm/bin/rocblas-bench",
        extra_arguments=["--solution_index", "44"],
    )
    assert "--solution_index 44" in pinned_text


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
    assert deltas_by_metric["macro_tile_x"]["baseline"] == 128
    assert deltas_by_metric["tensile_workgroup_size_y"]["baseline"] == 4
    assert deltas_by_metric["flat_workgroup_size"]["baseline"] == 128


def test_rocblas_shape_comparisons_match_linear_benchmark_config(
    tmp_path: Path,
) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    benchmark_path = tmp_path / "results.jsonl"
    _write_rocblas_log(rocblas_log_path)
    _write_single_benchmark_jsonl(
        benchmark_path,
        "loom_qkv",
        _linear_compile_report_payload(
            output_size=13824,
            token_count=4608,
            input_size=4608,
        ),
        27_582_000,
    )
    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("loom_qkv", benchmark_path)],
        rocblas_log_paths=[NamedPath("rocblas", rocblas_log_path)],
    )

    specs = build_kernel_anatomy_rocblas_benchmark_shape_comparison_specs(report)

    assert specs == [
        ComparisonSpec(
            baseline="rocblas/gemm_M13824_N4547_K4608_beta0",
            candidate="loom_qkv",
        )
    ]
    comparisons = build_kernel_anatomy_comparisons(report, specs)
    comparison = comparisons["rocblas/gemm_M13824_N4547_K4608_beta0=loom_qkv"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["operation_time_ns"]["ratio"] == 27582000 / 8503930.0
    assert (
        "config.benchmark.linear_wmma.output_size"
        in comparison["missing_baseline_metrics"]
    )


def test_rocblas_shape_comparisons_include_same_named_disassembly(
    tmp_path: Path,
) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    disassembly_path = tmp_path / "rocblas.s"
    benchmark_path = tmp_path / "results.jsonl"
    _write_rocblas_log(rocblas_log_path)
    _write(
        disassembly_path,
        """
0000000000000000 <selected_tensile>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
      ds_read_b64 v[0:1], v0
      ds_write_b64 v0, v[0:1]
""",
    )
    compile_report = _linear_compile_report_payload(
        output_size=13824,
        token_count=4608,
        input_size=4608,
    )
    entry = compile_report["entries"]["rows"][0]
    entry["static_instruction_mix"]["local_memory_count"] = 16
    entry["static_instruction_mix"]["local_read_byte_count"] = 64
    entry["static_instruction_mix"]["local_write_byte_count"] = 64
    _write_single_benchmark_jsonl(
        benchmark_path,
        "loom_qkv",
        compile_report,
        27_582_000,
    )
    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("rocblas", disassembly_path)],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("loom_qkv", benchmark_path)],
        rocblas_log_paths=[NamedPath("rocblas", rocblas_log_path)],
    )

    specs = build_kernel_anatomy_rocblas_benchmark_shape_comparison_specs(report)
    comparisons = build_kernel_anatomy_comparisons(report, specs)

    comparison = comparisons["rocblas/gemm_M13824_N4547_K4608_beta0=loom_qkv"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["operation_time_ns"]["ratio"] == 27582000 / 8503930.0
    assert deltas_by_metric["local_memory_instruction_count"]["baseline"] == 2
    assert deltas_by_metric["local_memory_instruction_count"]["candidate"] == 16
    assert deltas_by_metric["local_memory_instruction_count"]["baseline_source"] == (
        "rocblas_disassembly"
    )
    scorecard_by_metric = {entry["metric"]: entry for entry in comparison["scorecard"]}
    assert scorecard_by_metric["local_memory_instruction_count"]["category"] == (
        "local_memory"
    )


def test_best_candidate_rows_select_fastest_matching_shape(tmp_path: Path) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    slow_benchmark_path = tmp_path / "slow_results.jsonl"
    fast_benchmark_path = tmp_path / "fast_results.jsonl"
    _write_rocblas_log(rocblas_log_path)
    compile_report = _linear_compile_report_payload(
        output_size=13824,
        token_count=4608,
        input_size=4608,
    )
    _write_single_benchmark_jsonl(
        slow_benchmark_path,
        "loom_slow_qkv",
        compile_report,
        27_582_000,
    )
    _write_single_benchmark_jsonl(
        fast_benchmark_path,
        "loom_fast_qkv",
        compile_report,
        20_000_000,
    )
    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[
            NamedPath("loom_slow_qkv", slow_benchmark_path),
            NamedPath("loom_fast_qkv", fast_benchmark_path),
        ],
        rocblas_log_paths=[NamedPath("rocblas", rocblas_log_path)],
    )
    specs = build_kernel_anatomy_rocblas_benchmark_shape_comparison_specs(report)
    comparisons = build_kernel_anatomy_comparisons(report, specs)

    best_rows = build_kernel_anatomy_best_candidate_rows(comparisons)

    assert best_rows == [
        {
            "baseline": "rocblas/gemm_M13824_N4547_K4608_beta0",
            "candidate": "loom_fast_qkv",
            "comparison": "rocblas/gemm_M13824_N4547_K4608_beta0=loom_fast_qkv",
            "status": "slower_unexplained",
            "time_ratio": 20000000 / 8503930.0,
            "primary_cost": {},
            "primary_advantage": {},
            "candidate_count": 2,
        }
    ]


def test_main_emits_rocblas_shape_summary(tmp_path: Path, capsys) -> None:
    rocblas_log_path = tmp_path / "rocblas.log"
    benchmark_path = tmp_path / "results.jsonl"
    _write_rocblas_log(rocblas_log_path)
    _write_single_benchmark_jsonl(
        benchmark_path,
        "loom_qkv",
        _linear_compile_report_payload(
            output_size=13824,
            token_count=4608,
            input_size=4608,
        ),
        27_582_000,
    )

    assert (
        main(
            [
                "--rocblas-log",
                f"rocblas={rocblas_log_path}",
                "--benchmark-jsonl",
                f"loom_qkv={benchmark_path}",
                "--compare-rocblas-benchmark-shapes",
                "--format",
                "summary",
            ]
        )
        == 0
    )

    text = capsys.readouterr().out
    assert "rocblas/gemm_M13824_N4547_K4608_beta0=loom_qkv" in text
    assert "slower_unexplained" in text
    assert "time=3.24344x" in text


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
    assert deltas_by_metric["tensile_workgroup_size_x"]["candidate"] == 32
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
    assert "Gap findings:" in text
    assert "Comparison verdicts:" in text
    assert "baseline=loom: shared=2 baseline_metrics=2" in text
    assert "scorecard:" in text
    assert "local_memory_bytes [local_memory]: candidate_higher" in text
    assert "candidate_cost/candidate_higher: local_memory_bytes" in text
    assert "baseline=loom: structural_cost" in text
    assert "primary cost: local_memory_bytes [local_memory]" in text
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
    assert whole_deltas["dynamic_local_memory_read_bytes"]["baseline"] == 320
    assert whole_deltas["dynamic_local_memory_write_bytes"]["baseline"] == 160
    assert whole_deltas["dynamic_local_memory_access_bytes"]["baseline"] == 480
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
    assert (
        deltas_by_metric["local_memory_instruction_count_per_matrix_instruction"][
            "baseline"
        ]
        == 2
    )
    assert (
        deltas_by_metric["local_memory_instruction_count_per_matrix_instruction"][
            "candidate"
        ]
        == 4.95
    )
    assert deltas_by_metric["local_memory_access_bytes"]["baseline"] == 160
    assert (
        deltas_by_metric["local_memory_access_bytes_per_matrix_instruction"]["baseline"]
        == 32
    )
    assert deltas_by_metric["wmma_count"]["baseline"] == 5
    assert comparison["scorecard"][0]["metric"] == "local_memory_instruction_count"
    assert comparison["scorecard"][0]["category"] == "local_memory"
    assert comparison["scorecard"][0]["severity"] == 9.9
    aggregate_scorecard = build_kernel_anatomy_comparison_scorecard(comparisons)
    assert aggregate_scorecard[0]["comparison"] == (
        "tensile/weighted_symbols=loom/dynamic"
    )
    assert aggregate_scorecard[0]["metric"] == "local_memory_instruction_count"

    verdicts = build_kernel_anatomy_comparison_verdicts(comparisons)
    assert verdicts[0]["comparison"] == ("tensile/weighted_symbols=loom/dynamic")
    assert verdicts[0]["status"] == "structural_cost"
    assert verdicts[0]["primary_cost"]["metric"] == "local_memory_instruction_count"
    assert verdicts[0]["primary_cost"]["category"] == "local_memory"


def test_optimization_frontier_flags_unconverted_advantage(tmp_path: Path) -> None:
    baseline_report = _compile_report_payload()
    candidate_report = json.loads(json.dumps(_compile_report_payload()))
    candidate_mix = candidate_report["entries"]["rows"][0]["dynamic_instruction_mix"]
    candidate_mix["local_memory_count"] = 40
    candidate_mix["local_read_byte_count"] = 120
    candidate_mix["local_write_byte_count"] = 80
    _write_single_benchmark_jsonl(
        tmp_path / "baseline.jsonl",
        "baseline_benchmark",
        baseline_report,
        1_000_000,
    )
    _write_single_benchmark_jsonl(
        tmp_path / "candidate.jsonl",
        "candidate_benchmark",
        candidate_report,
        1_040_000,
    )
    _write_single_benchmark_jsonl(
        tmp_path / "discard.jsonl",
        "discard_benchmark",
        candidate_report,
        900_000,
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[
            NamedPath("baseline", tmp_path / "baseline.jsonl"),
            NamedPath("candidate", tmp_path / "candidate.jsonl"),
            NamedPath("discard", tmp_path / "discard.jsonl"),
        ],
    )
    frontier = build_kernel_anatomy_optimization_frontier(
        report, ["baseline"], ["candidate$"]
    )

    whole_kernel_entry = next(
        entry for entry in frontier if entry["candidate"] == "candidate"
    )
    assert [entry["candidate"] for entry in frontier] == ["candidate"]
    assert whole_kernel_entry["status"] == "advantage_not_converted"
    assert whole_kernel_entry["time_ratio"] == 1.04
    assert whole_kernel_entry["primary_advantage"]["category"] == "local_memory"
    assert whole_kernel_entry["candidate_metrics"]["p50_ns"] == 1_040_000

    report["optimization_frontier"] = frontier
    text = format_text_report(report)
    assert "Optimization frontier:" in text
    assert "baseline -> candidate: advantage_not_converted" in text


def test_weighted_symbol_group_metrics_participate_in_comparisons(
    tmp_path: Path,
) -> None:
    disassembly_path = tmp_path / "tensile.s"
    _write(
        disassembly_path,
        """
0000000000000000 <open_loop_a>:
      ds_read_b128 v[0:3], v0
      ds_write_b128 v0, v[0:3]
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000100 <open_loop_b>:
      ds_read_b128 v[0:3], v0
      ds_write_b128 v0, v[0:3]
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000200 <writeback>:
      buffer_store_b128 v0, v[0:3], s[0:3], 0 offen
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("tensile", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        weighted_symbol_group_specs=[
            WeightedSymbolGroupSpec("tensile", "open_loop", "open_loop_a", 3),
            WeightedSymbolGroupSpec("tensile", "open_loop", "open_loop_b", 2),
            WeightedSymbolGroupSpec("tensile", "writeback", "writeback", 1),
        ],
    )
    comparisons = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                baseline="tensile/weighted/open_loop",
                candidate="loom/dynamic",
            )
        ],
    )

    comparison = comparisons["tensile/weighted/open_loop=loom/dynamic"]
    deltas_by_metric = {delta["metric"]: delta for delta in comparison["deltas"]}
    assert deltas_by_metric["local_memory_instruction_count"]["baseline"] == 10
    assert deltas_by_metric["local_memory_instruction_count"]["candidate"] == 99
    assert (
        deltas_by_metric["local_memory_instruction_count_per_matrix_instruction"][
            "ratio"
        ]
        == 2.475
    )
    assert deltas_by_metric["wmma_count"]["baseline"] == 5
    assert comparison["scorecard"][0]["category"] == "local_memory"


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


def test_main_emits_benchmark_comparison_scorecard(tmp_path: Path, capsys) -> None:
    benchmark_path = tmp_path / "compare_only.jsonl"
    _write_compare_only_benchmark_jsonl(benchmark_path)

    assert (
        main(
            [
                "--benchmark-jsonl",
                f"bench={benchmark_path}",
                "--format",
                "text",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Comparison scorecard:" in text
    assert "Comparison verdicts:" in text
    assert "bench/c0/bench_kernel=bench/c1/bench_kernel" in text
    assert "operation_time_ns [time]: candidate_higher" in text


def test_summary_report_compacts_benchmark_comparison_verdicts(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "compare_only.jsonl"
    _write_compare_only_benchmark_jsonl(benchmark_path)
    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )
    report["comparisons"] = build_kernel_anatomy_comparisons(
        report,
        [
            ComparisonSpec(
                "bench/c0/bench_kernel",
                "bench/c1/bench_kernel",
            )
        ],
    )
    report["comparison_verdicts"] = build_kernel_anatomy_comparison_verdicts(
        report["comparisons"]
    )
    report["comparison_scorecard"] = build_kernel_anatomy_comparison_scorecard(
        report["comparisons"]
    )

    text = format_summary_report(report)

    assert "Kernel anatomy summary" in text
    assert "Benchmark bundles:" in text
    assert "Comparison verdicts:" in text
    assert "bench c0->c1 bench_kernel" in text
    assert "slower_with_structural_cost time=1.1x" in text
    assert "Top structural costs:" in text


def test_summary_report_flags_duplicate_benchmark_candidates(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "duplicate_candidates.jsonl"
    _write_duplicate_candidate_benchmark_jsonl(benchmark_path)
    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        benchmark_jsonl_paths=[NamedPath("bench", benchmark_path)],
    )

    text = format_summary_report(report)

    assert "Duplicate benchmark candidates:" in text
    assert "bench c0->c1 bench_kernel" in text
    assert "same_compile_signature time=1.007x" in text


def test_main_emits_summary_report(tmp_path: Path, capsys) -> None:
    benchmark_path = tmp_path / "compare_only.jsonl"
    _write_compare_only_benchmark_jsonl(benchmark_path)

    assert (
        main(
            [
                "--benchmark-jsonl",
                f"bench={benchmark_path}",
                "--format",
                "summary",
            ]
        )
        == 0
    )

    text = capsys.readouterr().out
    assert "Kernel anatomy summary" in text
    assert "bench c0->c1 bench_kernel" in text
    assert "slower_with_structural_cost time=1.1x" in text


def test_main_accepts_weighted_symbol_group(tmp_path: Path, capsys) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <open_loop>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )

    assert (
        main(
            [
                "--disassembly",
                f"asm={disassembly_path}",
                "--weighted-symbol-group",
                "asm/open_loop=open_loop=4",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    group = report["disassemblies"]["asm"]["weighted_symbol_groups"]["open_loop"]
    assert group["summary"]["family_counts"]["v_wmma"] == 4


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


def test_main_emits_rocblas_solution_trace_script(tmp_path: Path, capsys) -> None:
    rocblas_log_path = tmp_path / "rocblas_trace.log"
    _write_rocblas_trace_log(rocblas_log_path)

    assert (
        main(
            [
                "--rocblas-log",
                f"pytorch={rocblas_log_path}",
                "--rocblas-bench-executable",
                "/tools/rocblas-bench",
                "--format",
                "rocblas-solution-trace",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "TENSILE_SOLUTION_SELECTION_TRACE=1" in text
    assert "/tools/rocblas-bench --function gemm_ex" in text
    assert "--cold_iters 0 --iters 1" in text


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
    assert comparison["shared_metric_count"] >= 3
    assert any(
        delta["metric"] == "local_memory_bytes" for delta in comparison["deltas"]
    )
    assert report["comparison_scorecard"][0]["comparison"] == "baseline=loom"
    assert report["comparison_verdicts"][0]["comparison"] == "baseline=loom"
    assert report["comparison_verdicts"][0]["status"] == "candidate_faster"


def test_main_emits_optimization_frontier(tmp_path: Path, capsys) -> None:
    baseline_report = _compile_report_payload()
    candidate_report = json.loads(json.dumps(_compile_report_payload()))
    candidate_mix = candidate_report["entries"]["rows"][0]["dynamic_instruction_mix"]
    candidate_mix["local_memory_count"] = 40
    candidate_mix["local_read_byte_count"] = 120
    candidate_mix["local_write_byte_count"] = 80
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    discard_path = tmp_path / "discard.jsonl"
    _write_single_benchmark_jsonl(
        baseline_path,
        "baseline_benchmark",
        baseline_report,
        1_000_000,
    )
    _write_single_benchmark_jsonl(
        candidate_path,
        "candidate_benchmark",
        candidate_report,
        1_040_000,
    )
    _write_single_benchmark_jsonl(
        discard_path,
        "discard_benchmark",
        candidate_report,
        900_000,
    )

    assert (
        main(
            [
                "--benchmark-jsonl",
                f"baseline={baseline_path}",
                "--benchmark-jsonl",
                f"candidate={candidate_path}",
                "--frontier",
                "baseline",
                "--frontier-candidate-regex",
                "candidate$",
                "--benchmark-jsonl",
                f"discard={discard_path}",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    whole_kernel_entry = next(
        entry
        for entry in report["optimization_frontier"]
        if entry["candidate"] == "candidate"
    )
    assert [entry["candidate"] for entry in report["optimization_frontier"]] == [
        "candidate"
    ]
    assert whole_kernel_entry["status"] == "advantage_not_converted"
    assert whole_kernel_entry["candidate_metrics"]["p50_ns"] == 1_040_000


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
