# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Audits the production Loom C/H source surface.

The audit intentionally counts source files, not build graph nodes. It is meant
to answer the product-shape question "how much Loom C/H ships in this stack?"
with explicit inclusion rules that can be compared over time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RUNTIME_PY_ROOT = Path(__file__).resolve().parents[2]
if str(_RUNTIME_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_PY_ROOT))

from loom.gen.support.generated_file import GENERATED_FILE_MARKER  # noqa: E402

SCHEMA_VERSION = 1
SOURCE_ROOT = Path("loom/src/loom")
SOURCE_SUFFIXES = (".c", ".h")

EXCLUDED_PATH_SEGMENTS = (
    ".notes",
    "cmake",
    "editor",
    "gen",
    "generator",
    "generators",
    "test",
    "testing",
    "tool",
    "tooling",
    "tools",
)

EXCLUDED_FILE_STEM_PREFIXES = ("test_",)

EXCLUDED_FILE_STEM_SUFFIXES = (
    "_test",
    "_test_util",
)

EXCLUSION_RULES = (
    {
        "kind": "path_segment",
        "values": EXCLUDED_PATH_SEGMENTS,
        "reason": (
            "Drops tests, harnesses, developer tools, editor metadata, "
            "CMake helpers, and local notes."
        ),
    },
    {
        "kind": "file_stem_prefix",
        "values": EXCLUDED_FILE_STEM_PREFIXES,
        "reason": (
            "Drops test-only support files that live beside production target code."
        ),
    },
    {
        "kind": "file_stem_suffix",
        "values": EXCLUDED_FILE_STEM_SUFFIXES,
        "reason": (
            "Drops test binaries, test registries, and test utility headers "
            "outside test directories."
        ),
    },
)

LINE_METRIC_DESCRIPTIONS = {
    "physical_lines": "Source line records, counting a final unterminated line.",
    "nonblank_lines": "Physical lines whose stripped text is non-empty.",
}


@dataclass(frozen=True)
class LineCounts:
    physical_lines: int
    nonblank_lines: int


@dataclass(frozen=True)
class SourceFileStats:
    relative_path: Path
    generated: bool
    line_counts: LineCounts


@dataclass(frozen=True)
class SourceSlice:
    name: str
    description: str
    includes: Callable[[Path], bool]


ALL_TARGETS_SLICE = SourceSlice(
    name="all_targets",
    description=(
        "All production Loom loom/src/loom .c/.h files after excluding tests, "
        "harnesses, tools, generators, editor metadata, and local-only files."
    ),
    includes=lambda relative_path: True,
)


def _is_native_amdgpu_hsaco_path(relative_path: Path) -> bool:
    parts = relative_path.parts
    if not parts:
        return False
    if parts[0] != "target":
        return True
    if len(parts) == 1:
        return True

    target_area = parts[1]
    if target_area == "arch":
        return len(parts) >= 3 and parts[2] == "amdgpu"
    if target_area == "emit":
        if len(parts) >= 3 and parts[2] == "native":
            return len(parts) == 4 or (len(parts) >= 5 and parts[3] == "amdgpu")
        return False
    return len(parts) == 2


NATIVE_AMDGPU_HSACO_SLICE = SourceSlice(
    name="native_amdgpu_hsaco",
    description=(
        "Production Loom core plus native AMDGPU/HSACO target code. This keeps "
        "target root support, target/arch/amdgpu, target/emit/native common "
        "code, and target/emit/native/amdgpu, while excluding optional module "
        "emitters, x86, aggregate target registries, and test/tool targets."
    ),
    includes=_is_native_amdgpu_hsaco_path,
)

SOURCE_SLICES = (
    ALL_TARGETS_SLICE,
    NATIVE_AMDGPU_HSACO_SLICE,
)


def _has_excluded_segment(relative_path: Path) -> bool:
    return any(part in EXCLUDED_PATH_SEGMENTS for part in relative_path.parts)


def _has_excluded_file_stem(relative_path: Path) -> bool:
    stem = relative_path.stem
    return any(
        stem.startswith(prefix) for prefix in EXCLUDED_FILE_STEM_PREFIXES
    ) or any(stem.endswith(suffix) for suffix in EXCLUDED_FILE_STEM_SUFFIXES)


def is_production_source_path(relative_path: Path) -> bool:
    return (
        relative_path.suffix in SOURCE_SUFFIXES
        and not _has_excluded_segment(relative_path)
        and not _has_excluded_file_stem(relative_path)
    )


def _count_lines(text: str) -> LineCounts:
    physical_lines = text.count("\n")
    if text and not text.endswith("\n"):
        physical_lines += 1
    nonblank_lines = sum(1 for line in text.splitlines() if line.strip())
    return LineCounts(
        physical_lines=physical_lines,
        nonblank_lines=nonblank_lines,
    )


def _read_source_file(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    header = "\n".join(text.splitlines()[:16])
    return text, header


def _scan_source_file(source_root: Path, path: Path) -> SourceFileStats:
    relative_path = path.relative_to(source_root)
    text, header = _read_source_file(path)
    return SourceFileStats(
        relative_path=relative_path,
        generated=GENERATED_FILE_MARKER in header,
        line_counts=_count_lines(text),
    )


def discover_source_files(source_root: Path) -> list[SourceFileStats]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing Loom source root: {source_root}")
    paths = (
        path
        for path in source_root.rglob("*")
        if path.is_file() and is_production_source_path(path.relative_to(source_root))
    )
    return [_scan_source_file(source_root, path) for path in sorted(paths)]


def _summarize_files(files: Sequence[SourceFileStats]) -> dict[str, int]:
    generated_files = [file_stats for file_stats in files if file_stats.generated]
    return {
        "files": len(files),
        "generated_files": len(generated_files),
        "generated_nonblank_lines": sum(
            file.line_counts.nonblank_lines for file in generated_files
        ),
        "generated_physical_lines": sum(
            file.line_counts.physical_lines for file in generated_files
        ),
        "nonblank_lines": sum(file.line_counts.nonblank_lines for file in files),
        "physical_lines": sum(file.line_counts.physical_lines for file in files),
    }


def _summarize_slice(
    source_slice: SourceSlice, files: Sequence[SourceFileStats]
) -> dict[str, Any]:
    slice_files = [
        file_stats
        for file_stats in files
        if source_slice.includes(file_stats.relative_path)
    ]
    hand_authored_files = [
        file_stats for file_stats in slice_files if not file_stats.generated
    ]
    return {
        "description": source_slice.description,
        "generated_inclusive": _summarize_files(slice_files),
        "hand_authored_only": _summarize_files(hand_authored_files),
    }


def build_source_surface_report(repo_root: Path) -> dict[str, Any]:
    source_root = repo_root / SOURCE_ROOT
    source_files = discover_source_files(source_root)
    return {
        "schema": "loom.source_surface",
        "schema_version": SCHEMA_VERSION,
        "source_root": SOURCE_ROOT.as_posix(),
        "source_suffixes": list(SOURCE_SUFFIXES),
        "line_metrics": LINE_METRIC_DESCRIPTIONS,
        "exclusion_rules": [
            {
                "kind": rule["kind"],
                "reason": rule["reason"],
                "values": list(rule["values"]),
            }
            for rule in EXCLUSION_RULES
        ],
        "slices": {
            source_slice.name: _summarize_slice(source_slice, source_files)
            for source_slice in SOURCE_SLICES
        },
    }


def _format_count(value: int) -> str:
    return f"{value:,}"


def format_text_report(report: dict[str, Any]) -> str:
    rows: list[tuple[str, str, dict[str, int]]] = []
    for slice_name, slice_report in report["slices"].items():
        rows.append(
            (slice_name, "generated_inclusive", slice_report["generated_inclusive"])
        )
        rows.append(
            (slice_name, "hand_authored_only", slice_report["hand_authored_only"])
        )

    header = (
        f"{'slice':<22} {'mode':<22} {'files':>7} {'generated':>9} "
        f"{'physical':>10} {'nonblank':>10}"
    )
    lines = [
        "Loom source-surface audit",
        f"schema: {report['schema']}.v{report['schema_version']}",
        f"source_root: {report['source_root']}",
        "",
        header,
        "-" * len(header),
    ]
    for slice_name, mode, summary in rows:
        lines.append(
            f"{slice_name:<22} {mode:<22} "
            f"{_format_count(summary['files']):>7} "
            f"{_format_count(summary['generated_files']):>9} "
            f"{_format_count(summary['physical_lines']):>10} "
            f"{_format_count(summary['nonblank_lines']):>10}"
        )
    lines.extend(
        (
            "",
            "Line metrics:",
            f"  physical_lines: {report['line_metrics']['physical_lines']}",
            f"  nonblank_lines: {report['line_metrics']['nonblank_lines']}",
            "",
            "Exclusion rules:",
        )
    )
    for rule in report["exclusion_rules"]:
        values = ", ".join(rule["values"])
        lines.append(f"  {rule['kind']}: {values} ({rule['reason']})")
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Repository root to audit. Defaults to the root containing this tool.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format. The JSON format is the stable snapshot format.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    try:
        report = build_source_surface_report(args.repo_root.resolve())
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if args.format == "json":
        sys.stdout.write(f"{json.dumps(report, indent=2, sort_keys=True)}\n")
    else:
        sys.stdout.write(f"{format_text_report(report)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
