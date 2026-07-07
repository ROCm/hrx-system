#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compares two joined ID4 dispatch profile reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispatch_profile_join import DispatchProfileJoinError


@dataclass(frozen=True)
class AggregateKey:
    stage_key: str
    name: str
    module_path: str
    function_name: str


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DispatchProfileJoinError(f"{field_name} must be an object")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise DispatchProfileJoinError(f"{field_name} must be a list")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DispatchProfileJoinError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise DispatchProfileJoinError(f"{field_name} must be an integer")
    return value


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    value = _require_int(value, field_name)
    if value < 0:
        raise DispatchProfileJoinError(f"{field_name} must be non-negative")
    return value


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DispatchProfileJoinError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise DispatchProfileJoinError(f"invalid {description}: {path}: {exc}") from exc
    return _require_dict(payload, description)


def _profile_duration_suffix(report: dict[str, Any]) -> str:
    summary = _require_dict(report.get("summary"), "summary")
    if summary.get("total_duration_ns") is not None:
        return "ns"
    if summary.get("total_duration_ticks") is not None:
        return "ticks"
    raise DispatchProfileJoinError("profile report has no comparable duration total")


def _duration_field_name(prefix: str, suffix: str) -> str:
    if suffix == "ns":
        return f"{prefix}_duration_ns"
    if suffix == "ticks":
        return f"{prefix}_duration_ticks"
    raise DispatchProfileJoinError(f"unsupported duration suffix: {suffix}")


def _duration_value(row: dict[str, Any], prefix: str, suffix: str) -> int | None:
    return _optional_nonnegative_int(
        row.get(_duration_field_name(prefix, suffix)),
        _duration_field_name(prefix, suffix),
    )


def _aggregate_key(row: dict[str, Any], aggregate_name: str) -> AggregateKey:
    stage_key = row.get("stage_key")
    if stage_key is None:
        stage_key = ""
    else:
        stage_key = _require_string(stage_key, f"{aggregate_name}.stage_key")
    name = row.get("name")
    if name is None:
        name = row.get("module_path")
    if name is None:
        name = row.get("function_name")
    return AggregateKey(
        stage_key=stage_key,
        name=_require_string(name, f"{aggregate_name}.name"),
        module_path=_require_string(
            row.get("module_path"), f"{aggregate_name}.module_path"
        ),
        function_name=_require_string(
            row.get("function_name"), f"{aggregate_name}.function_name"
        ),
    )


def _aggregate_rows_by_key(
    rows: list[Any], aggregate_name: str
) -> dict[AggregateKey, dict[str, Any]]:
    rows_by_key = {}
    for index, value in enumerate(rows):
        row = _require_dict(value, f"{aggregate_name}[{index}]")
        key = _aggregate_key(row, aggregate_name)
        if key in rows_by_key:
            raise DispatchProfileJoinError(
                f"{aggregate_name} contains duplicate aggregate key {key}"
            )
        rows_by_key[key] = row
    return rows_by_key


def _format_delta(candidate_value: int | None, base_value: int | None) -> str:
    if candidate_value is None or base_value is None:
        return "n/a"
    delta = candidate_value - base_value
    if delta > 0:
        return f"+{delta}"
    return str(delta)


def _format_ratio(candidate_value: int | None, base_value: int | None) -> str:
    if candidate_value is None or base_value is None or base_value == 0:
        return "n/a"
    return f"{candidate_value / base_value:.3f}x"


def _format_change_percent(candidate_value: int | None, base_value: int | None) -> str:
    if candidate_value is None or base_value is None or base_value == 0:
        return "n/a"
    return f"{100.0 * (candidate_value - base_value) / base_value:+.2f}%"


def _format_duration(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _comparison_sort_key(
    key: AggregateKey,
    base_by_key: dict[AggregateKey, dict[str, Any]],
    candidate_by_key: dict[AggregateKey, dict[str, Any]],
    duration_suffix: str,
) -> tuple[int, str, str, str]:
    base_duration = _duration_value(base_by_key.get(key, {}), "total", duration_suffix)
    candidate_duration = _duration_value(
        candidate_by_key.get(key, {}), "total", duration_suffix
    )
    return (
        -max(base_duration or 0, candidate_duration or 0),
        key.stage_key,
        key.name,
        key.function_name,
    )


def _format_count(
    candidate_row: dict[str, Any],
    base_row: dict[str, Any],
) -> str:
    candidate_count = _optional_nonnegative_int(candidate_row.get("count"), "count")
    base_count = _optional_nonnegative_int(base_row.get("count"), "count")
    if candidate_count is None and base_count is None:
        return "n/a"
    return f"{candidate_count or 0}/{base_count or 0}"


def _compare_rows(
    base_rows: list[Any],
    candidate_rows: list[Any],
    aggregate_name: str,
    duration_suffix: str,
    limit: int,
) -> list[str]:
    base_by_key = _aggregate_rows_by_key(base_rows, aggregate_name)
    candidate_by_key = _aggregate_rows_by_key(candidate_rows, aggregate_name)
    keys = sorted(
        set(base_by_key) | set(candidate_by_key),
        key=lambda key: _comparison_sort_key(
            key, base_by_key, candidate_by_key, duration_suffix
        ),
    )
    lines = [
        f"## {aggregate_name}",
        "| Candidate | Base | Delta | Ratio | Change | Count | Name | Function |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for key in keys[:limit]:
        base_row = base_by_key.get(key, {})
        candidate_row = candidate_by_key.get(key, {})
        base_duration = _duration_value(base_row, "total", duration_suffix)
        candidate_duration = _duration_value(candidate_row, "total", duration_suffix)
        name = key.name if not key.stage_key else f"{key.stage_key}:{key.name}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_duration(candidate_duration),
                    _format_duration(base_duration),
                    _format_delta(candidate_duration, base_duration),
                    _format_ratio(candidate_duration, base_duration),
                    _format_change_percent(candidate_duration, base_duration),
                    _format_count(candidate_row, base_row),
                    name,
                    key.function_name,
                ]
            )
            + " |"
        )
    return lines


def format_comparison_report(
    base_report: dict[str, Any],
    candidate_report: dict[str, Any],
    limit: int = 20,
) -> str:
    """Returns a Markdown comparison between two joined profile reports."""

    if limit <= 0:
        raise DispatchProfileJoinError("comparison limit must be positive")
    base_suffix = _profile_duration_suffix(base_report)
    candidate_suffix = _profile_duration_suffix(candidate_report)
    if base_suffix != candidate_suffix:
        raise DispatchProfileJoinError(
            f"profile duration units differ: base={base_suffix} candidate={candidate_suffix}"
        )
    base_summary = _require_dict(base_report.get("summary"), "base.summary")
    candidate_summary = _require_dict(
        candidate_report.get("summary"), "candidate.summary"
    )
    base_total = _duration_value(base_summary, "total", base_suffix)
    candidate_total = _duration_value(candidate_summary, "total", candidate_suffix)
    lines = [
        "# Dispatch Profile Comparison",
        "",
        f"duration_unit={candidate_suffix}",
        (
            f"candidate_total={_format_duration(candidate_total)} "
            f"base_total={_format_duration(base_total)} "
            f"delta={_format_delta(candidate_total, base_total)} "
            f"ratio={_format_ratio(candidate_total, base_total)} "
            f"change={_format_change_percent(candidate_total, base_total)}"
        ),
        "",
    ]
    lines.extend(
        _compare_rows(
            _require_list(base_report.get("by_operation"), "base.by_operation"),
            _require_list(
                candidate_report.get("by_operation"), "candidate.by_operation"
            ),
            "Operations",
            candidate_suffix,
            limit,
        )
    )
    lines.append("")
    lines.extend(
        _compare_rows(
            _require_list(base_report.get("by_kernel"), "base.by_kernel"),
            _require_list(candidate_report.get("by_kernel"), "candidate.by_kernel"),
            "Kernels",
            candidate_suffix,
            limit,
        )
    )
    lines.append("")
    return "\n".join(lines)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Baseline joined profile JSON path.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Candidate joined profile JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown comparison output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows per comparison section.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_arguments(argv)
    try:
        base_report = _load_json_object(args.base, "base profile report")
        candidate_report = _load_json_object(args.candidate, "candidate profile report")
        comparison = format_comparison_report(
            base_report, candidate_report, limit=args.limit
        )
        if args.output is None:
            sys.stdout.write(comparison)
            sys.stdout.write("\n")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(comparison, encoding="utf-8")
    except DispatchProfileJoinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
