# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiled execution economics for compile report views."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportError

_MISSING = object()


@dataclass(frozen=True)
class _ExecutionMetricSpec:
    """One fixed-trip metric derived from the target-low instruction mix."""

    key: str
    label: str
    mix_field: str | None
    economics_group: str
    economics_field: str
    unit: str | None = None
    show_zero: bool = False


_EXECUTION_METRIC_SPECS = (
    _ExecutionMetricSpec(
        "scalar_alu_count",
        "scalar ALU operation effects",
        "scalar_alu_count",
        "operations",
        "scalar_alu_count",
        show_zero=True,
    ),
    _ExecutionMetricSpec(
        "vector_alu_count",
        "vector ALU operation effects",
        "vector_alu_count",
        "operations",
        "vector_alu_count",
        show_zero=True,
    ),
    _ExecutionMetricSpec(
        "matrix_count",
        "matrix operation effects",
        "matrix_count",
        "operations",
        "matrix_count",
        show_zero=True,
    ),
    _ExecutionMetricSpec(
        "mfma_count",
        "MFMA operation effects",
        "mfma_count",
        "operations",
        "mfma_count",
    ),
    _ExecutionMetricSpec(
        "smfmac_count",
        "SMFMAC operation effects",
        "smfmac_count",
        "operations",
        "smfmac_count",
    ),
    _ExecutionMetricSpec(
        "wmma_count",
        "WMMA operation effects",
        "wmma_count",
        "operations",
        "wmma_count",
    ),
    _ExecutionMetricSpec(
        "swmmac_count",
        "SWMMAC operation effects",
        "swmmac_count",
        "operations",
        "swmmac_count",
    ),
    _ExecutionMetricSpec(
        "dot_count",
        "dot operation effects",
        "dot_count",
        "operations",
        "dot_count",
    ),
    _ExecutionMetricSpec(
        "atomic_count",
        "atomic operation effects",
        "atomic_count",
        "operations",
        "atomic_count",
    ),
    _ExecutionMetricSpec(
        "branch_count",
        "branch operation effects",
        "branch_count",
        "operations",
        "branch_count",
    ),
    _ExecutionMetricSpec(
        "barrier_count",
        "barrier operation effects",
        "barrier_count",
        "operations",
        "barrier_count",
    ),
    _ExecutionMetricSpec(
        "control_count",
        "control operation effects",
        "control_count",
        "operations",
        "control_count",
    ),
    _ExecutionMetricSpec(
        "conversion_count",
        "conversion operation effects",
        "conversion_count",
        "operations",
        "conversion_count",
    ),
    _ExecutionMetricSpec(
        "cache_count",
        "cache operation effects",
        "cache_count",
        "operations",
        "cache_count",
    ),
    _ExecutionMetricSpec(
        "register_move_count",
        "register-move operation effects",
        "register_move_count",
        "operations",
        "register_move_count",
    ),
    _ExecutionMetricSpec(
        "global_load_count",
        "global-load operation effects",
        "global_load_count",
        "memory",
        "global_load_count",
    ),
    _ExecutionMetricSpec(
        "global_store_count",
        "global-store operation effects",
        "global_store_count",
        "memory",
        "global_store_count",
    ),
    _ExecutionMetricSpec(
        "buffer_load_count",
        "buffer-load operation effects",
        "buffer_load_count",
        "memory",
        "buffer_load_count",
    ),
    _ExecutionMetricSpec(
        "buffer_store_count",
        "buffer-store operation effects",
        "buffer_store_count",
        "memory",
        "buffer_store_count",
    ),
    _ExecutionMetricSpec(
        "flat_memory_count",
        "flat-memory operation effects",
        "flat_memory_count",
        "memory",
        "flat_memory_count",
    ),
    _ExecutionMetricSpec(
        "local_memory_count",
        "local-memory operation effects",
        "local_memory_count",
        "memory",
        "local_memory_count",
    ),
    _ExecutionMetricSpec(
        "scalar_memory_count",
        "scalar-memory operation effects",
        "scalar_memory_count",
        "memory",
        "scalar_memory_count",
    ),
    _ExecutionMetricSpec(
        "private_memory_count",
        "private-memory operation effects",
        "private_memory_count",
        "memory",
        "private_memory_count",
    ),
    _ExecutionMetricSpec(
        "generic_memory_count",
        "generic-memory operation effects",
        "generic_memory_count",
        "memory",
        "generic_memory_count",
    ),
    _ExecutionMetricSpec(
        "issued_read_bytes",
        "issued read widths",
        "memory_read_byte_count",
        "memory",
        "read_bytes",
        unit="bytes",
        show_zero=True,
    ),
    _ExecutionMetricSpec(
        "issued_write_bytes",
        "issued write widths",
        "memory_write_byte_count",
        "memory",
        "write_bytes",
        unit="bytes",
        show_zero=True,
    ),
    _ExecutionMetricSpec(
        "issued_total_bytes",
        "total issued widths",
        None,
        "memory",
        "total_bytes",
        unit="bytes",
        show_zero=True,
    ),
    _ExecutionMetricSpec(
        "issued_read_unknown_width_count",
        "reads with unknown issued width",
        "memory_read_unknown_width_count",
        "memory",
        "read_unknown_width_count",
    ),
    _ExecutionMetricSpec(
        "issued_write_unknown_width_count",
        "writes with unknown issued width",
        "memory_write_unknown_width_count",
        "memory",
        "write_unknown_width_count",
    ),
)

_UNAVAILABLE_REASONS = {
    "exact_fixed_trip_multiplicities_unavailable": (
        "exact fixed-trip multiplicities were not proven"
    ),
}


def build_execution_economics_show(
    entry: dict[str, object],
    workload: dict[str, object],
    source: str,
) -> dict[str, object]:
    """Builds reliable compiled economics without requiring dispatch geometry."""
    dynamic_mix_value = entry.get("dynamic_instruction_mix")
    if dynamic_mix_value is None:
        return _unavailable_economics("exact_fixed_trip_multiplicities_unavailable")
    dynamic_mix = _require_object(
        dynamic_mix_value, f"{source}.dynamic_instruction_mix"
    )

    per_workitem_metrics: dict[str, object] = {}
    for spec in _EXECUTION_METRIC_SPECS:
        if spec.mix_field is None:
            continue
        per_workitem_metrics[spec.key] = _require_unsigned_integer(
            dynamic_mix.get(spec.mix_field),
            f"{source}.dynamic_instruction_mix.{spec.mix_field}",
        )
    per_workitem_metrics["issued_total_bytes"] = (
        per_workitem_metrics["issued_read_bytes"]
        + per_workitem_metrics["issued_write_bytes"]
    )
    _validate_retained_economics(entry, workload, per_workitem_metrics, source)

    scopes: dict[str, object] = {
        "workitem": {
            "metrics": per_workitem_metrics,
        }
    }
    primary_scope = "workitem"
    workitems_per_workgroup = workload.get("workitems_per_workgroup")
    if isinstance(workitems_per_workgroup, int) and not isinstance(
        workitems_per_workgroup, bool
    ):
        if workitems_per_workgroup == 0:
            raise CompileReportError(
                f"{source}.workload.workgroup_size: expected a nonzero product"
            )
        scopes["workgroup"] = {
            "workitem_count": workitems_per_workgroup,
            "metrics": _scale_metrics(per_workitem_metrics, workitems_per_workgroup),
        }
        primary_scope = "workgroup"

    unknown_width_count = (
        per_workitem_metrics["issued_read_unknown_width_count"]
        + per_workitem_metrics["issued_write_unknown_width_count"]
    )
    return {
        "fixed_trip_multiplicity_coverage": "exact",
        "control_flow_basis": "statically_reachable_block_envelope",
        "operation_unit": "target_low_operation_effects",
        "issued_byte_coverage": "partial" if unknown_width_count else "exact",
        "issued_byte_basis": "descriptor_effect_widths_not_memory_transactions",
        "primary_scope": primary_scope,
        "scopes": scopes,
    }


def build_execution_economics_diff(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Builds a deterministic diff between compiled economics views."""
    baseline_coverage = baseline["fixed_trip_multiplicity_coverage"]
    candidate_coverage = candidate["fixed_trip_multiplicity_coverage"]
    baseline_byte_coverage = baseline["issued_byte_coverage"]
    candidate_byte_coverage = candidate["issued_byte_coverage"]
    comparison_scope, scope_workitem_count = _select_comparison_scope(
        baseline, candidate
    )
    baseline_metrics = _scope_metrics(baseline, comparison_scope, "baseline")
    candidate_metrics = _scope_metrics(candidate, comparison_scope, "candidate")

    changed: dict[str, object] = {}
    incomplete: dict[str, object] = {}
    unchanged_count = 0
    unavailable_count = 0
    for spec in _EXECUTION_METRIC_SPECS:
        baseline_value = baseline_metrics.get(spec.key, _MISSING)
        candidate_value = candidate_metrics.get(spec.key, _MISSING)
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
        metric_diff: dict[str, object] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
        delta = candidate_value - baseline_value
        metric_diff["delta"] = delta
        if baseline_value != 0:
            metric_diff["change_percent"] = delta * 100.0 / baseline_value
        changed[spec.key] = metric_diff

    view: dict[str, object] = {
        "fixed_trip_multiplicity_coverage": {
            "baseline": baseline_coverage,
            "candidate": candidate_coverage,
        },
        "issued_byte_coverage": {
            "baseline": baseline_byte_coverage,
            "candidate": candidate_byte_coverage,
        },
        "comparison_scope": comparison_scope,
        "changed": changed,
        "incomplete": incomplete,
        "unchanged_count": unchanged_count,
        "unavailable_count": unavailable_count,
    }
    if scope_workitem_count is not None:
        view["scope_workitem_count"] = scope_workitem_count
    baseline_reason = baseline.get("reason")
    candidate_reason = candidate.get("reason")
    if baseline_reason is not None or candidate_reason is not None:
        view["reasons"] = {
            "baseline": baseline_reason,
            "candidate": candidate_reason,
        }
    return view


def execution_economics_diff_has_changes(
    economics_diff: dict[str, object],
) -> bool:
    """Returns whether coverage or any compiled-economics metric changed."""
    fixed_trip_coverage = _require_object(
        economics_diff["fixed_trip_multiplicity_coverage"],
        "fixed_trip_multiplicity_coverage",
    )
    byte_coverage = _require_object(
        economics_diff["issued_byte_coverage"], "issued_byte_coverage"
    )
    reasons = _require_object(economics_diff.get("reasons", {}), "reasons")
    return bool(
        fixed_trip_coverage["baseline"] != fixed_trip_coverage["candidate"]
        or byte_coverage["baseline"] != byte_coverage["candidate"]
        or reasons.get("baseline") != reasons.get("candidate")
        or economics_diff["changed"]
        or economics_diff["incomplete"]
    )


def append_execution_economics_show_text(
    lines: list[str], economics: dict[str, object]
) -> None:
    """Appends compiled execution economics to a human-readable report view."""
    lines.append("  Compiled execution economics (compiler analysis)")
    coverage = economics["fixed_trip_multiplicity_coverage"]
    lines.append(f"    fixed-trip multiplicity coverage: {coverage}")
    if coverage != "exact":
        lines.append(f"    reason: {_format_reason(economics.get('reason'))}")
        return
    lines.append(
        "    control-flow basis: all statically reachable blocks (path envelope)"
    )
    primary_scope = str(economics["primary_scope"])
    scopes = _require_object(economics["scopes"], "economics.scopes")
    scope = _require_object(scopes[primary_scope], f"economics.{primary_scope}")
    if primary_scope == "workgroup":
        lines.append(
            "    scope: one workgroup "
            f"({_format_number(scope['workitem_count'])} workitems)"
        )
    else:
        lines.append("    scope: one workitem")
    lines.append("    operation unit: target-low operation effects")
    lines.append(f"    issued byte coverage: {economics['issued_byte_coverage']}")
    lines.append("    issued byte basis: descriptor-effect widths, not transactions")
    metrics = _require_object(scope["metrics"], "economics.metrics")
    omitted_zero_count = 0
    for spec in _EXECUTION_METRIC_SPECS:
        value = metrics[spec.key]
        if value == 0 and not spec.show_zero:
            omitted_zero_count += 1
            continue
        lines.append(f"    {spec.label}: {_format_value(value, spec.unit)}")
    if omitted_zero_count:
        lines.append(f"    zero-valued categories omitted: {omitted_zero_count}")


def append_execution_economics_diff_text(
    lines: list[str], economics_diff: dict[str, object]
) -> None:
    """Appends changed compiled economics to a human-readable report diff."""
    lines.append("  Compiled execution economics (compiler analysis)")
    fixed_trip_coverage = _require_object(
        economics_diff["fixed_trip_multiplicity_coverage"],
        "fixed_trip_multiplicity_coverage",
    )
    lines.append(
        "    fixed-trip multiplicity coverage: "
        f"{_format_comparison(fixed_trip_coverage)}"
    )
    reasons = _require_object(economics_diff.get("reasons", {}), "reasons")
    if reasons:
        lines.append(f"    reason: {_format_reason_comparison(reasons)}")
    if (
        fixed_trip_coverage["baseline"] == "exact"
        or fixed_trip_coverage["candidate"] == "exact"
    ):
        comparison_scope = economics_diff["comparison_scope"]
        if comparison_scope == "workgroup":
            lines.append(
                "    comparison scope: one workgroup "
                f"({_format_number(economics_diff['scope_workitem_count'])} "
                "workitems)"
            )
        else:
            lines.append("    comparison scope: one workitem")
        lines.append("    operation unit: target-low operation effects")
        byte_coverage = _require_object(
            economics_diff["issued_byte_coverage"], "issued_byte_coverage"
        )
        lines.append(f"    issued byte coverage: {_format_comparison(byte_coverage)}")
        lines.append(
            "    issued byte basis: descriptor-effect widths, not transactions"
        )

    changed = _require_object(economics_diff["changed"], "economics.changed")
    incomplete = _require_object(economics_diff["incomplete"], "economics.incomplete")
    if not changed and not incomplete:
        lines.append("    no metric changes")
    for spec in _EXECUTION_METRIC_SPECS:
        metric_value = changed.get(spec.key, incomplete.get(spec.key))
        if metric_value is None:
            continue
        metric = _require_object(metric_value, f"economics.{spec.key}")
        baseline = _format_value(metric["baseline"], spec.unit)
        candidate = _format_value(metric["candidate"], spec.unit)
        suffix = ""
        if "delta" in metric:
            suffix = f", delta {_format_signed_value(metric['delta'], spec.unit)}"
        if "change_percent" in metric:
            suffix += f" ({metric['change_percent']:+.2f}%)"
        lines.append(f"    {spec.label}: {baseline} -> {candidate}{suffix}")
    lines.append(
        f"    unchanged: {economics_diff['unchanged_count']}; "
        f"unavailable in both: {economics_diff['unavailable_count']}"
    )


def _unavailable_economics(reason: str) -> dict[str, object]:
    return {
        "fixed_trip_multiplicity_coverage": "unavailable",
        "issued_byte_coverage": "unavailable",
        "reason": reason,
    }


def _scale_metrics(
    per_workitem_metrics: dict[str, object], scale: int
) -> dict[str, object]:
    return {key: value * scale for key, value in per_workitem_metrics.items()}


def _select_comparison_scope(
    baseline: dict[str, object], candidate: dict[str, object]
) -> tuple[str, int | None]:
    baseline_scopes = _require_object(baseline.get("scopes", {}), "baseline.scopes")
    candidate_scopes = _require_object(candidate.get("scopes", {}), "candidate.scopes")
    baseline_workgroup = baseline_scopes.get("workgroup")
    candidate_workgroup = candidate_scopes.get("workgroup")
    if isinstance(baseline_workgroup, dict) and isinstance(candidate_workgroup, dict):
        baseline_count = baseline_workgroup.get("workitem_count")
        candidate_count = candidate_workgroup.get("workitem_count")
        if baseline_count == candidate_count and isinstance(baseline_count, int):
            return "workgroup", baseline_count
    return "workitem", None


def _scope_metrics(
    economics: dict[str, object], scope: str, source: str
) -> dict[str, object]:
    scopes = _require_object(economics.get("scopes", {}), f"{source}.scopes")
    scope_value = scopes.get(scope)
    if scope_value is None:
        return {}
    scope_object = _require_object(scope_value, f"{source}.scopes.{scope}")
    return _require_object(
        scope_object.get("metrics", {}), f"{source}.scopes.{scope}.metrics"
    )


def _validate_retained_economics(
    entry: dict[str, object],
    workload: dict[str, object],
    per_workitem_metrics: dict[str, object],
    source: str,
) -> None:
    _validate_retained_scope(
        entry,
        per_workitem_metrics,
        source,
        operation_scope="per_workitem",
        memory_scope="per_workitem_issued",
    )
    has_dispatch_economics = (
        _lookup(entry, "economics.operations.dispatch") is not _MISSING
        or _lookup(entry, "economics.memory.dispatch_issued") is not _MISSING
    )
    if not has_dispatch_economics:
        return
    dispatch_workitem_count = workload.get("dispatch_workitem_count")
    if not isinstance(dispatch_workitem_count, int) or isinstance(
        dispatch_workitem_count, bool
    ):
        raise CompileReportError(
            f"{source}.economics: dispatch totals require a static dispatch "
            "workitem count"
        )
    _validate_retained_scope(
        entry,
        _scale_metrics(per_workitem_metrics, dispatch_workitem_count),
        source,
        operation_scope="dispatch",
        memory_scope="dispatch_issued",
    )


def _validate_retained_scope(
    entry: dict[str, object],
    expected_metrics: dict[str, object],
    source: str,
    *,
    operation_scope: str,
    memory_scope: str,
) -> None:
    scope_paths = {
        "operations": f"economics.operations.{operation_scope}",
        "memory": f"economics.memory.{memory_scope}",
    }
    for group, path in scope_paths.items():
        retained_value = _lookup(entry, path)
        if retained_value is _MISSING:
            continue
        retained = _require_object(retained_value, f"{source}.{path}")
        for spec in _EXECUTION_METRIC_SPECS:
            if spec.economics_group != group:
                continue
            value = retained.get(spec.economics_field, _MISSING)
            if value is _MISSING:
                continue
            retained_count = _require_unsigned_integer(
                value, f"{source}.{path}.{spec.economics_field}"
            )
            if retained_count != expected_metrics[spec.key]:
                raise CompileReportError(
                    f"{source}.{path}.{spec.economics_field}: expected "
                    f"{expected_metrics[spec.key]}, got {retained_count}"
                )


def _lookup(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _format_comparison(comparison: dict[str, object]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    if baseline == candidate:
        return f"{baseline} in both"
    return f"{baseline} -> {candidate}"


def _format_reason_comparison(reasons: dict[str, object]) -> str:
    baseline = reasons.get("baseline")
    candidate = reasons.get("candidate")
    if baseline == candidate:
        return f"{_format_reason(baseline)} in both"
    return f"{_format_reason(baseline)} -> {_format_reason(candidate)}"


def _format_reason(reason: object) -> str:
    if reason is None:
        return "none"
    return _UNAVAILABLE_REASONS.get(str(reason), str(reason))


def _format_value(value: object, unit: str | None) -> str:
    if value is None:
        return "unavailable"
    formatted = _format_number(value)
    return f"{formatted} B" if unit == "bytes" else formatted


def _format_signed_value(value: object, unit: str | None) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        formatted = f"{value:+,}"
    elif isinstance(value, float):
        formatted = f"{value:+.4g}"
    else:
        formatted = str(value)
    return f"{formatted} B" if unit == "bytes" else formatted


def _format_number(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{source}: expected an object")
    return value


def _require_unsigned_integer(value: object, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CompileReportError(f"{source}: expected an unsigned integer")
    return value
