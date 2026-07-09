# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Summarizes target kernel artifacts into comparable anatomy reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.tools.amdgpu_asm import (
    AmdgpuDisassemblyBlock,
    summarize_amdgpu_disassembly,
    summarize_amdgpu_disassembly_blocks,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NamedPath:
    name: str
    path: Path


def _parse_named_path(value: str) -> NamedPath:
    if "=" not in value:
        path = Path(value)
        return NamedPath(name=path.stem, path=path)
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("named paths must have a non-empty name")
    return NamedPath(name=name, path=Path(path))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_compile_report(data: Any) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError("compile report root must be a JSON object")
    report = data.get("compile_report", data)
    if not isinstance(report, Mapping):
        raise ValueError("compile_report must be a JSON object")
    return report


def _first_entry(report: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = report.get("entries")
    if not isinstance(entries, Mapping):
        return report
    rows = entries.get("rows")
    if not isinstance(rows, list) or not rows:
        return report
    entry = rows[0]
    if not isinstance(entry, Mapping):
        return report
    return entry


def _summarize_loom_compile_report(
    report: Mapping[str, Any], path: Path | None = None
) -> dict[str, Any]:
    """Extracts kernel-economics fields from a Loom compile-report object."""

    entry = _first_entry(report)
    target_resources = entry.get("target_resources")
    if not isinstance(target_resources, Mapping):
        target_resources = {}
    static_instruction_mix = entry.get("static_instruction_mix")
    if not isinstance(static_instruction_mix, Mapping):
        static_instruction_mix = {}
    dynamic_instruction_mix = entry.get("dynamic_instruction_mix")
    if not isinstance(dynamic_instruction_mix, Mapping):
        dynamic_instruction_mix = {}
    economics = entry.get("economics")
    if not isinstance(economics, Mapping):
        economics = {}
    workload = entry.get("workload")
    if not isinstance(workload, Mapping):
        workload = {}
    wait_plan = entry.get("wait_plan")
    if not isinstance(wait_plan, Mapping):
        wait_plan = {}
    summary = {
        "function": report.get("function") or entry.get("function"),
        "target_key": report.get("target_key"),
        "executable_format": report.get("executable_format"),
        "workload": workload,
        "instruction_count": entry.get("instruction_count"),
        "code_byte_count": entry.get("code_byte_count"),
        "private_memory_bytes": entry.get("private_memory_bytes"),
        "local_memory_bytes": entry.get("local_memory_bytes"),
        "static_instruction_mix": static_instruction_mix,
        "dynamic_instruction_mix": dynamic_instruction_mix,
        "economics": economics,
        "target_resources": target_resources,
        "wait_plan": wait_plan,
    }
    if path is not None:
        summary["path"] = path.as_posix()
    return summary


def summarize_loom_compile_report(path: Path) -> dict[str, Any]:
    """Extracts kernel-economics fields from a Loom compile-report artifact."""

    report = _extract_compile_report(_read_json(path))
    return _summarize_loom_compile_report(report, path)


def summarize_loom_benchmark_jsonl(path: Path) -> dict[str, Any]:
    """Extracts timing and compile summaries from benchmark JSONL events."""

    benchmark_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if not isinstance(event, Mapping) or event.get("row") != "benchmark":
            continue
        benchmark_result = event.get("benchmark_result")
        if not isinstance(benchmark_result, Mapping):
            continue
        measurement = benchmark_result.get("measurement")
        if not isinstance(measurement, Mapping):
            measurement = {}
        operation_timing = measurement.get("operation_timing_ns")
        if not isinstance(operation_timing, Mapping):
            operation_timing = {}
        timing_interpretation = measurement.get("timing_interpretation")
        if not isinstance(timing_interpretation, Mapping):
            timing_interpretation = {}
        correctness = benchmark_result.get("correctness")
        if not isinstance(correctness, Mapping):
            correctness = {}
        compile_report = benchmark_result.get("compile_report")
        compile_summary = None
        if isinstance(compile_report, Mapping):
            compile_summary = _summarize_loom_compile_report(compile_report)
        benchmark_rows.append(
            {
                "benchmark": benchmark_result.get("benchmark"),
                "case": benchmark_result.get("case"),
                "state": benchmark_result.get("state"),
                "candidate_id": event.get("candidate_id"),
                "candidate_index": event.get("candidate_index"),
                "sample_compilation": benchmark_result.get("sample_compilation"),
                "timing_ns": operation_timing,
                "timing_interpretation": timing_interpretation,
                "correctness": correctness,
                "compile_report": compile_summary,
            }
        )
    return {
        "path": path.as_posix(),
        "benchmark_count": len(benchmark_rows),
        "benchmarks": benchmark_rows,
    }


_KERNEL_BLOCK_PATTERN = re.compile(r"^\s+- \.args:\s*$")
_KERNEL_FIELD_PATTERN = re.compile(r"^    \.(\w+):\s*(.*)$")
_INTEGER_FIELDS = frozenset(
    {
        "group_segment_fixed_size",
        "kernarg_segment_align",
        "kernarg_segment_size",
        "max_flat_workgroup_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "wavefront_size",
    }
)


def summarize_amdhsa_metadata_path(
    named_path: NamedPath,
    metadata_regex: re.Pattern[str] | None,
    top_kernel_count: int,
) -> dict[str, Any]:
    """Extracts kernel resource metadata from `llvm-readobj --notes` text."""

    kernels = tuple(
        kernel
        for kernel in _iter_amdhsa_metadata_kernels(_read_text(named_path.path))
        if _matches_metadata_regex(kernel, metadata_regex)
    )
    return {
        "name": named_path.name,
        "path": named_path.path.as_posix(),
        "kernel_count": len(kernels),
        "kernels": list(kernels[: max(top_kernel_count, 0)]),
    }


def summarize_disassembly_path(
    named_path: NamedPath,
    symbol_regex: re.Pattern[str] | None,
    top_symbol_count: int,
    ordered_symbol_count: int,
) -> dict[str, Any]:
    """Builds a JSON-serializable disassembly summary for one artifact."""

    text = _read_text(named_path.path)
    whole_file = summarize_amdgpu_disassembly(text)
    blocks = summarize_amdgpu_disassembly_blocks(text)
    if symbol_regex is not None:
        blocks = tuple(block for block in blocks if symbol_regex.search(block.symbol))
    top_blocks = _top_blocks(blocks, top_symbol_count)
    return {
        "name": named_path.name,
        "path": named_path.path.as_posix(),
        "whole_file": whole_file.metadata(),
        "symbol_count": len(blocks),
        "top_symbols": [block.metadata() for block in top_blocks],
        "ordered_symbols": [
            block.metadata() for block in _ordered_blocks(blocks, ordered_symbol_count)
        ],
    }


def build_kernel_anatomy_report(
    disassembly_paths: Sequence[NamedPath],
    compile_report_paths: Sequence[NamedPath],
    benchmark_jsonl_paths: Sequence[NamedPath] = (),
    symbol_regex: str | None = None,
    amdhsa_metadata_paths: Sequence[NamedPath] = (),
    amdhsa_metadata_regex: str | None = None,
    top_symbol_count: int = 16,
    ordered_symbol_count: int = 0,
) -> dict[str, Any]:
    """Builds a stable anatomy report for dumped target artifacts."""

    compiled_symbol_regex = re.compile(symbol_regex) if symbol_regex else None
    compiled_metadata_regex = (
        re.compile(amdhsa_metadata_regex) if amdhsa_metadata_regex else None
    )
    compile_reports = {
        named_path.name: summarize_loom_compile_report(named_path.path)
        for named_path in compile_report_paths
    }
    benchmark_jsonl = {
        named_path.name: summarize_loom_benchmark_jsonl(named_path.path)
        for named_path in benchmark_jsonl_paths
    }
    amdhsa_metadata = {
        named_path.name: summarize_amdhsa_metadata_path(
            named_path, compiled_metadata_regex, top_symbol_count
        )
        for named_path in amdhsa_metadata_paths
    }
    disassemblies = {
        named_path.name: summarize_disassembly_path(
            named_path,
            compiled_symbol_regex,
            top_symbol_count,
            ordered_symbol_count,
        )
        for named_path in disassembly_paths
    }
    return {
        "schema": "loom.kernel_anatomy",
        "schema_version": SCHEMA_VERSION,
        "amdhsa_metadata": amdhsa_metadata,
        "disassemblies": disassemblies,
        "loom_benchmarks": benchmark_jsonl,
        "loom_compile_reports": compile_reports,
    }


def _iter_amdhsa_metadata_kernels(text: str) -> Sequence[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    current_kernel: dict[str, Any] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _KERNEL_BLOCK_PATTERN.match(line):
            if current_kernel is not None:
                kernels.append(current_kernel)
            current_kernel = {"start_line": line_number}
            continue
        if current_kernel is None:
            continue
        match = _KERNEL_FIELD_PATTERN.match(line)
        if match is None:
            continue
        key = match.group(1)
        value = match.group(2).strip()
        if key in _INTEGER_FIELDS:
            try:
                current_kernel[key] = int(value, 0)
            except ValueError:
                current_kernel[key] = value
        elif key in {"name", "symbol"}:
            current_kernel[key] = value
    if current_kernel is not None:
        kernels.append(current_kernel)
    return tuple(kernels)


def _matches_metadata_regex(
    kernel: Mapping[str, Any], metadata_regex: re.Pattern[str] | None
) -> bool:
    if metadata_regex is None:
        return True
    name = kernel.get("name")
    symbol = kernel.get("symbol")
    name_matches = isinstance(name, str) and metadata_regex.search(name) is not None
    symbol_matches = (
        isinstance(symbol, str) and metadata_regex.search(symbol) is not None
    )
    return name_matches or symbol_matches


def _top_blocks(
    blocks: Sequence[AmdgpuDisassemblyBlock], top_symbol_count: int
) -> tuple[AmdgpuDisassemblyBlock, ...]:
    if top_symbol_count <= 0:
        return ()
    return tuple(
        sorted(
            blocks,
            key=lambda block: (
                _matrix_count(block),
                block.summary.instruction_count,
                block.symbol,
            ),
            reverse=True,
        )[:top_symbol_count]
    )


def _ordered_blocks(
    blocks: Sequence[AmdgpuDisassemblyBlock], ordered_symbol_count: int
) -> tuple[AmdgpuDisassemblyBlock, ...]:
    if ordered_symbol_count <= 0:
        return ()
    return tuple(
        sorted(
            blocks,
            key=lambda block: (
                block.address is None,
                block.address if block.address is not None else block.start_line,
            ),
        )[:ordered_symbol_count]
    )


def _matrix_count(block: AmdgpuDisassemblyBlock) -> int:
    return sum(block.summary.matrix_mnemonic_counts.values())


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _append_compile_report_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    compile_reports = report.get("loom_compile_reports", {})
    if not isinstance(compile_reports, Mapping) or not compile_reports:
        return
    lines.append("")
    lines.append("Loom compile reports:")
    for name, compile_report_value in compile_reports.items():
        compile_report = _as_mapping(compile_report_value)
        resources = _as_mapping(compile_report.get("target_resources"))
        vector = _as_mapping(resources.get("vector"))
        vector_final = _as_mapping(vector.get("final"))
        mix = _as_mapping(compile_report.get("static_instruction_mix"))
        lines.append(
            "  "
            f"{name}: instructions={compile_report.get('instruction_count')} "
            f"code_bytes={compile_report.get('code_byte_count')} "
            f"local_bytes={compile_report.get('local_memory_bytes')} "
            f"vgpr={vector_final.get('register_count')} "
            f"occupancy={resources.get('occupancy_percent')}% "
            f"wmma={mix.get('wmma_count')} "
            f"mfma={mix.get('mfma_count')} "
            f"valu={mix.get('vector_alu_count')}"
        )


def _format_p50_ms(timing: Mapping[str, Any]) -> str:
    p50 = timing.get("p50")
    return f"{p50 / 1_000_000:.6g}" if isinstance(p50, (int, float)) else "?"


def _append_benchmark_report_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    benchmark_reports = report.get("loom_benchmarks", {})
    if not isinstance(benchmark_reports, Mapping) or not benchmark_reports:
        return
    lines.append("")
    lines.append("Loom benchmark JSONL:")
    for name, benchmark_report_value in benchmark_reports.items():
        benchmark_report = _as_mapping(benchmark_report_value)
        lines.append(f"  {name}: benchmarks={benchmark_report.get('benchmark_count')}")
        benchmarks = benchmark_report.get("benchmarks", [])
        if not isinstance(benchmarks, list):
            continue
        for benchmark_value in benchmarks[:8]:
            benchmark = _as_mapping(benchmark_value)
            timing = _as_mapping(benchmark.get("timing_ns"))
            correctness = _as_mapping(benchmark.get("correctness"))
            compile_report = _as_mapping(benchmark.get("compile_report"))
            resources = _as_mapping(compile_report.get("target_resources"))
            vector = _as_mapping(resources.get("vector"))
            vector_final = _as_mapping(vector.get("final"))
            static_mix = _as_mapping(compile_report.get("static_instruction_mix"))
            dynamic_mix = _as_mapping(compile_report.get("dynamic_instruction_mix"))
            lines.append(
                "    "
                f"{benchmark.get('benchmark')}: "
                f"state={benchmark.get('state')} "
                f"p50_ms={_format_p50_ms(timing)} "
                f"correctness={correctness.get('failed_sample_count')}/"
                f"{correctness.get('sample_count')} "
                f"instructions={compile_report.get('instruction_count')} "
                f"code_bytes={compile_report.get('code_byte_count')} "
                f"local_bytes={compile_report.get('local_memory_bytes')} "
                f"vgpr={vector_final.get('register_count')} "
                f"occupancy={resources.get('occupancy_percent')}% "
                f"wmma={static_mix.get('wmma_count')} "
                f"valu={static_mix.get('vector_alu_count')} "
                f"dynamic_local={dynamic_mix.get('local_memory_count')}"
            )


def _append_disassembly_symbol_lines(
    lines: list[str], symbols: Any, indent: str
) -> None:
    if not isinstance(symbols, list):
        return
    for symbol_value in symbols[:8]:
        symbol = _as_mapping(symbol_value)
        summary = _as_mapping(symbol.get("summary"))
        symbol_families = _as_mapping(summary.get("family_counts"))
        lines.append(
            f"{indent}{symbol.get('symbol')}: "
            f"instructions={summary.get('instruction_count')} "
            f"wmma={symbol_families.get('v_wmma', 0)} "
            f"mfma={symbol_families.get('v_mfma', 0)} "
            f"global_load={symbol_families.get('global_load', 0)} "
            f"global_store={symbol_families.get('global_store', 0)} "
            f"buffer_load={symbol_families.get('buffer_load', 0)} "
            f"buffer_store={symbol_families.get('buffer_store', 0)} "
            f"ds_read={symbol_families.get('ds_read', 0)} "
            f"ds_write={symbol_families.get('ds_write', 0)}"
        )


def _append_ordered_symbol_lines(lines: list[str], ordered_symbols: Any) -> None:
    if not isinstance(ordered_symbols, list) or not ordered_symbols:
        return
    lines.append("    ordered symbols:")
    for symbol_value in ordered_symbols:
        symbol = _as_mapping(symbol_value)
        summary = _as_mapping(symbol.get("summary"))
        symbol_families = _as_mapping(summary.get("family_counts"))
        address = symbol.get("address")
        address_text = f"0x{address:x}" if isinstance(address, int) else "?"
        lines.append(
            "      "
            f"{address_text} {symbol.get('symbol')}: "
            f"instructions={summary.get('instruction_count')} "
            f"wmma={symbol_families.get('v_wmma', 0)} "
            f"mfma={symbol_families.get('v_mfma', 0)} "
            f"buffer_load={symbol_families.get('buffer_load', 0)} "
            f"buffer_store={symbol_families.get('buffer_store', 0)} "
            f"ds_read={symbol_families.get('ds_read', 0)} "
            f"ds_write={symbol_families.get('ds_write', 0)} "
            f"branch={symbol_families.get('s_branch', 0)}"
        )


def _append_disassembly_report_lines(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    disassemblies = report.get("disassemblies", {})
    if not isinstance(disassemblies, Mapping) or not disassemblies:
        return
    lines.append("")
    lines.append("Disassemblies:")
    for name, disassembly_value in disassemblies.items():
        disassembly = _as_mapping(disassembly_value)
        whole_file = _as_mapping(disassembly.get("whole_file"))
        families = _as_mapping(whole_file.get("family_counts"))
        memory = _as_mapping(whole_file.get("memory_byte_counts"))
        lines.append(
            "  "
            f"{name}: symbols={disassembly.get('symbol_count')} "
            f"instructions={whole_file.get('instruction_count')} "
            f"wmma={families.get('v_wmma', 0)} "
            f"mfma={families.get('v_mfma', 0)} "
            f"global_load={families.get('global_load', 0)} "
            f"global_store={families.get('global_store', 0)} "
            f"buffer_load={families.get('buffer_load', 0)} "
            f"buffer_store={families.get('buffer_store', 0)} "
            f"ds_read={families.get('ds_read', 0)} "
            f"ds_write={families.get('ds_write', 0)} "
            f"wait={families.get('s_waitcnt', 0)} "
            f"barrier={families.get('s_barrier', 0)} "
            f"read_bytes={memory.get('read_bytes', 0)} "
            f"write_bytes={memory.get('write_bytes', 0)}"
        )
        _append_disassembly_symbol_lines(lines, disassembly.get("top_symbols"), "    ")
        _append_ordered_symbol_lines(lines, disassembly.get("ordered_symbols"))


def _append_amdhsa_metadata_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    amdhsa_metadata = report.get("amdhsa_metadata", {})
    if not isinstance(amdhsa_metadata, Mapping) or not amdhsa_metadata:
        return
    lines.append("")
    lines.append("AMDHSA metadata:")
    for name, metadata_value in amdhsa_metadata.items():
        metadata = _as_mapping(metadata_value)
        lines.append(f"  {name}: kernels={metadata.get('kernel_count')}")
        kernels = metadata.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels[:8]:
            kernel = _as_mapping(kernel_value)
            symbol = kernel.get("symbol") or kernel.get("name")
            lines.append(
                "    "
                f"{symbol}: "
                f"lds={kernel.get('group_segment_fixed_size')} "
                f"private={kernel.get('private_segment_fixed_size')} "
                f"vgpr={kernel.get('vgpr_count')} "
                f"sgpr={kernel.get('sgpr_count')} "
                f"vgpr_spills={kernel.get('vgpr_spill_count')} "
                f"sgpr_spills={kernel.get('sgpr_spill_count')} "
                f"wavefront={kernel.get('wavefront_size')} "
                f"workgroup={kernel.get('max_flat_workgroup_size')} "
                f"kernarg={kernel.get('kernarg_segment_size')}"
            )


def format_text_report(report: Mapping[str, Any]) -> str:
    """Formats the anatomy report as concise human-readable text."""

    lines = ["Kernel anatomy report"]
    _append_compile_report_lines(lines, report)
    _append_benchmark_report_lines(lines, report)
    _append_disassembly_report_lines(lines, report)
    _append_amdhsa_metadata_lines(lines, report)
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disassembly",
        action="append",
        default=[],
        type=_parse_named_path,
        help="Disassembly artifact as NAME=PATH, or PATH to use the stem as NAME.",
    )
    parser.add_argument(
        "--compile-report",
        action="append",
        default=[],
        type=_parse_named_path,
        help="Loom compile report JSON as NAME=PATH, or PATH to use the stem.",
    )
    parser.add_argument(
        "--benchmark-jsonl",
        action="append",
        default=[],
        type=_parse_named_path,
        help=("iree-benchmark-loom JSONL as NAME=PATH, or PATH to use the stem."),
    )
    parser.add_argument(
        "--amdhsa-metadata",
        action="append",
        default=[],
        type=_parse_named_path,
        help=(
            "`llvm-readobj --notes` output as NAME=PATH, or PATH to use the "
            "stem as NAME."
        ),
    )
    parser.add_argument(
        "--amdhsa-metadata-regex",
        help="Regex filter applied to AMDHSA kernel name and symbol metadata.",
    )
    parser.add_argument(
        "--symbol-regex",
        help="Regex filter applied to per-symbol disassembly summaries.",
    )
    parser.add_argument(
        "--top-symbols",
        type=int,
        default=16,
        help="Number of top symbol summaries to include per disassembly.",
    )
    parser.add_argument(
        "--ordered-symbols",
        type=int,
        default=0,
        help=(
            "Number of address-ordered symbol and label summaries to include "
            "per disassembly."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    report = build_kernel_anatomy_report(
        disassembly_paths=args.disassembly,
        compile_report_paths=args.compile_report,
        benchmark_jsonl_paths=args.benchmark_jsonl,
        symbol_regex=args.symbol_regex,
        amdhsa_metadata_paths=args.amdhsa_metadata,
        amdhsa_metadata_regex=args.amdhsa_metadata_regex,
        top_symbol_count=args.top_symbols,
        ordered_symbol_count=args.ordered_symbols,
    )
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
