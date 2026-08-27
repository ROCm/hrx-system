# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiled launch geometry views for Loom compile reports."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportError

_MISSING = object()


@dataclass(frozen=True)
class _WorkloadFieldSpec:
    """One ordered field in the workload show and diff views."""

    key: str
    label: str


_WORKLOAD_FIELD_SPECS = (
    _WorkloadFieldSpec("workgroup_size", "workgroup size"),
    _WorkloadFieldSpec("workitems_per_workgroup", "workitems per workgroup"),
    _WorkloadFieldSpec("workgroup_count", "workgroup count"),
    _WorkloadFieldSpec("dispatch_workgroup_count", "dispatch workgroups"),
    _WorkloadFieldSpec("cluster_size", "workgroup cluster size"),
    _WorkloadFieldSpec("subgroup_size", "subgroup size (workitems)"),
    _WorkloadFieldSpec("subgroups_per_workgroup", "subgroups per workgroup"),
    _WorkloadFieldSpec("dispatch_subgroup_count", "dispatch subgroups"),
    _WorkloadFieldSpec("dispatch_workitem_count", "dispatch workitems"),
)


def build_workload_show(
    workload_value: object,
    target_resources_value: object,
    source: str,
) -> dict[str, object]:
    """Builds compiled launch facts retained by one report record."""
    if workload_value is None:
        return {}
    workload = _require_object(workload_value, f"{source}.workload")
    workgroup_size = _parse_dimension(workload, "workgroup_size", source)
    workgroup_count = _parse_dimension(workload, "workgroup_count", source)
    cluster_size = _parse_dimension(workload, "cluster_size", source)

    view: dict[str, object] = {}
    if workgroup_size is not None:
        view["workgroup_size"] = workgroup_size
        if "flat" in workgroup_size:
            view["workitems_per_workgroup"] = workgroup_size["flat"]
    if workgroup_count is not None:
        view["workgroup_count"] = workgroup_count
        if "flat" in workgroup_count:
            view["dispatch_workgroup_count"] = workgroup_count["flat"]
    if cluster_size is not None:
        view["cluster_size"] = cluster_size

    subgroup_size = _parse_subgroup_size(target_resources_value, source)
    if subgroup_size is not None:
        view["subgroup_size"] = subgroup_size
        workitems_per_workgroup = view.get("workitems_per_workgroup")
        if isinstance(workitems_per_workgroup, int):
            subgroups_per_workgroup = (
                workitems_per_workgroup + subgroup_size - 1
            ) // subgroup_size
            view["subgroups_per_workgroup"] = subgroups_per_workgroup
            dispatch_workgroup_count = view.get("dispatch_workgroup_count")
            if isinstance(dispatch_workgroup_count, int):
                view["dispatch_subgroup_count"] = (
                    subgroups_per_workgroup * dispatch_workgroup_count
                )

    if "dispatch_workitem_count" in workload:
        dispatch_workitem_count = _require_unsigned_integer(
            workload["dispatch_workitem_count"],
            f"{source}.workload.dispatch_workitem_count",
        )
        workitems_per_workgroup = view.get("workitems_per_workgroup")
        dispatch_workgroup_count = view.get("dispatch_workgroup_count")
        if isinstance(workitems_per_workgroup, int) and isinstance(
            dispatch_workgroup_count, int
        ):
            expected_count = workitems_per_workgroup * dispatch_workgroup_count
            if dispatch_workitem_count != expected_count:
                raise CompileReportError(
                    f"{source}.workload.dispatch_workitem_count: expected "
                    f"{expected_count}, got {dispatch_workitem_count}"
                )
        view["dispatch_workitem_count"] = dispatch_workitem_count
    return view


def build_workload_diff(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Builds a deterministic diff between two normalized workload views."""
    changed: dict[str, object] = {}
    incomplete: dict[str, object] = {}
    unchanged_count = 0
    unavailable_count = 0
    for spec in _WORKLOAD_FIELD_SPECS:
        baseline_value = baseline.get(spec.key, _MISSING)
        candidate_value = candidate.get(spec.key, _MISSING)
        if baseline_value is _MISSING and candidate_value is _MISSING:
            unavailable_count += 1
            continue
        if baseline_value is _MISSING or candidate_value is _MISSING:
            incomplete[spec.key] = {
                "baseline": None if baseline_value is _MISSING else baseline_value,
                "candidate": None if candidate_value is _MISSING else candidate_value,
            }
            continue
        if baseline_value == candidate_value:
            unchanged_count += 1
            continue
        field_diff: dict[str, object] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
        if _is_number(baseline_value) and _is_number(candidate_value):
            delta = candidate_value - baseline_value
            field_diff["delta"] = delta
            if baseline_value != 0:
                field_diff["change_percent"] = delta * 100.0 / baseline_value
        changed[spec.key] = field_diff
    return {
        "changed": changed,
        "incomplete": incomplete,
        "unchanged_count": unchanged_count,
        "unavailable_count": unavailable_count,
    }


def workload_diff_has_changes(workload_diff: dict[str, object] | None) -> bool:
    """Returns whether a workload diff contains a changed or incomplete fact."""
    return workload_diff is not None and bool(
        workload_diff["changed"] or workload_diff["incomplete"]
    )


def append_workload_show_text(
    lines: list[str],
    workload: dict[str, object],
    *,
    indent: str = "",
) -> None:
    """Appends compiled launch facts to a human-readable view."""
    lines.append(f"{indent}Launch geometry (compiled artifact)")
    if not workload:
        lines.append(f"{indent}  static launch geometry: unavailable")
        return
    for spec in _WORKLOAD_FIELD_SPECS:
        value = workload.get(spec.key, _MISSING)
        lines.append(f"{indent}  {spec.label}: {_format_value(value, spec.key)}")


def append_workload_diff_text(
    lines: list[str],
    workload_diff: dict[str, object],
    *,
    indent: str = "",
) -> None:
    """Appends changed compiled launch facts to a human-readable diff."""
    changed = _require_object(workload_diff.get("changed"), "workload.changed")
    incomplete = _require_object(workload_diff.get("incomplete"), "workload.incomplete")
    lines.append(f"{indent}Launch geometry (compiled artifact)")
    if not changed and not incomplete:
        lines.append(f"{indent}  no launch geometry changes")
    for spec in _WORKLOAD_FIELD_SPECS:
        field_value = changed.get(spec.key, incomplete.get(spec.key))
        if field_value is None:
            continue
        field_diff = _require_object(field_value, f"workload.{spec.key}")
        baseline = _format_value(field_diff["baseline"], spec.key)
        candidate = _format_value(field_diff["candidate"], spec.key)
        suffix = ""
        if "delta" in field_diff:
            suffix = f", delta {_format_signed_value(field_diff['delta'])}"
        if "change_percent" in field_diff:
            suffix += f" ({field_diff['change_percent']:+.2f}%)"
        lines.append(f"{indent}  {spec.label}: {baseline} -> {candidate}{suffix}")
    lines.append(
        f"{indent}  unchanged: {workload_diff['unchanged_count']}; "
        f"unavailable in both: {workload_diff['unavailable_count']}"
    )


def _parse_dimension(
    workload: dict[str, object], key: str, source: str
) -> dict[str, object] | None:
    value = workload.get(key)
    if value is None:
        return None
    dimension = _require_object(value, f"{source}.workload.{key}")
    parsed = {
        axis: _require_unsigned_integer(
            dimension.get(axis), f"{source}.workload.{key}.{axis}"
        )
        for axis in ("x", "y", "z")
    }
    if "flat" in dimension:
        flat = _require_unsigned_integer(
            dimension["flat"], f"{source}.workload.{key}.flat"
        )
        expected_flat = parsed["x"] * parsed["y"] * parsed["z"]
        if flat != expected_flat:
            raise CompileReportError(
                f"{source}.workload.{key}.flat: expected {expected_flat}, got {flat}"
            )
        parsed["flat"] = flat
    return parsed


def _parse_subgroup_size(target_resources_value: object, source: str) -> int | None:
    if target_resources_value is None:
        return None
    target_resources = _require_object(
        target_resources_value, f"{source}.target_resources"
    )
    if "subgroup_size" not in target_resources:
        return None
    subgroup_size = _require_unsigned_integer(
        target_resources["subgroup_size"],
        f"{source}.target_resources.subgroup_size",
    )
    return subgroup_size or None


def _format_value(value: object, key: str) -> str:
    if value is _MISSING or value is None:
        return "unavailable"
    if key in ("workgroup_size", "workgroup_count", "cluster_size"):
        dimension = _require_object(value, key)
        return f"({dimension['x']}, {dimension['y']}, {dimension['z']})"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _format_signed_value(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:+,}"
    if isinstance(value, float):
        return f"{value:+.4g}"
    return str(value)


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{source}: expected object")
    return value


def _require_unsigned_integer(value: object, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompileReportError(f"{source}: expected unsigned integer")
    return value
