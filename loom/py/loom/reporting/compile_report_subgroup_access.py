# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Subgroup memory-access views, semantic diffs, and text formatting."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportDocument, CompileReportError

_MISSING = object()


@dataclass(frozen=True)
class _MetricSpec:
    """One stable subgroup-access summary scalar."""

    key: str
    label: str
    path: str


_METRIC_SPECS = (
    _MetricSpec("modeled_packet_count", "modeled packets", "modeled_packet_count"),
    _MetricSpec("exact_packet_count", "exact packets", "exact_packet_count"),
    _MetricSpec("unknown_packet_count", "unknown packets", "unknown_packet_count"),
    _MetricSpec(
        "dense_packet_count",
        "dense packets",
        "structural.dense_packet_count",
    ),
    _MetricSpec(
        "gapped_packet_count",
        "gapped packets",
        "structural.gapped_packet_count",
    ),
    _MetricSpec(
        "overlapping_packet_count",
        "overlapping packets",
        "structural.overlapping_packet_count",
    ),
    _MetricSpec(
        "exact_dynamic_packet_count",
        "source packets with exact geometry/count contributions",
        "dynamic.exact_packet_count",
    ),
    _MetricSpec(
        "unknown_dynamic_packet_count",
        "source packets without exact geometry/count contributions",
        "dynamic.unknown_packet_count",
    ),
    _MetricSpec(
        "dynamic_packet_count",
        "packet executions with exact geometry/count contributions",
        "dynamic.packet_count",
    ),
    _MetricSpec(
        "dynamic_dense_packet_count",
        "dynamic dense packet executions",
        "dynamic.dense_packet_count",
    ),
    _MetricSpec(
        "dynamic_gapped_packet_count",
        "dynamic gapped packet executions",
        "dynamic.gapped_packet_count",
    ),
    _MetricSpec(
        "dynamic_overlapping_packet_count",
        "dynamic overlapping packet executions",
        "dynamic.overlapping_packet_count",
    ),
)

_GROUP_IDENTITY_FIELDS = (
    "function",
    "source_op",
    "source_op_kind",
    "source_root",
    "source_root_argument_index",
    "memory_space",
    "operation",
    "packet",
    "strategy",
)

_SOURCE_IDENTITY_FIELDS = _GROUP_IDENTITY_FIELDS[:-2]

_INTEGER_IDENTITY_FIELDS = (
    "source_op_kind",
    "source_root_argument_index",
)

_ADDRESS_STRING_FIELDS = (
    "lane_address_proof",
    "active_lane_proof",
    "lane_mapping",
)

_ADDRESS_INTEGER_FIELDS = (
    "subgroup_size",
    "per_lane_packet_bytes",
    "linear_lane_stride_bytes",
)

_GEOMETRY_INTEGER_FIELDS = (
    "subgroup_requested_bytes",
    "subgroup_unique_bytes",
    "subgroup_span_bytes",
    "maximum_adjacent_lane_delta_bytes",
    "maximum_uncovered_gap_bytes",
    "distinct_lane_address_count",
    "contiguous_region_count",
)


def build_subgroup_access_show(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds the compact subgroup-access view when evidence is available."""
    memory = _optional_memory(document)
    if memory is None:
        return None
    summary = memory.get("subgroup_access")
    groups = memory.get("subgroup_access_groups")
    if summary is None and groups is None:
        return None
    summary_object = _require_object(
        summary,
        f"{document.source}.source_low.memory.subgroup_access",
    )
    group_values = _require_list(
        groups,
        f"{document.source}.source_low.memory.subgroup_access_groups",
    )
    expected_group_count = memory.get("subgroup_access_group_count")
    if (
        not isinstance(expected_group_count, int)
        or isinstance(expected_group_count, bool)
        or expected_group_count != len(group_values)
    ):
        raise CompileReportError(
            f"{document.source}.source_low.memory.subgroup_access_group_count: "
            f"expected {len(group_values)}, got {expected_group_count!r}"
        )

    shown_summary = _show_metrics(
        summary_object,
        f"{document.source}.source_low.memory.subgroup_access",
    )
    shown_groups = []
    for index, value in enumerate(group_values):
        group_source = (
            f"{document.source}.source_low.memory.subgroup_access_groups[{index}]"
        )
        group = _show_group(
            value,
            group_source,
        )
        _validate_group_target_subgroup_size(document, group, group_source)
        shown_groups.append(group)
    shown_groups.sort(key=_group_sort_key)
    _validate_unique_variants(shown_groups, document.source)
    group_summary = _sum_group_summaries(shown_groups)
    if group_summary != shown_summary:
        raise CompileReportError(
            f"{document.source}.source_low.memory.subgroup_access: summary does "
            "not equal the subgroup_access_groups partition"
        )
    return {
        "summary": shown_summary,
        "group_count": len(shown_groups),
        "groups": shown_groups,
    }


def build_subgroup_access_diff(
    baseline: CompileReportDocument,
    candidate: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds a source-aware diff without pairing ambiguous packet variants."""
    baseline_show = build_subgroup_access_show(baseline)
    candidate_show = build_subgroup_access_show(candidate)
    if baseline_show is None and candidate_show is None:
        return None
    if baseline_show is None or candidate_show is None:
        return {
            "availability": {
                "baseline": "available" if baseline_show is not None else "unavailable",
                "candidate": (
                    "available" if candidate_show is not None else "unavailable"
                ),
            },
            "changed_source_count": 0,
            "unchanged_source_count": 0,
            "sources": [],
        }

    summary_diff = _diff_metrics(
        _expect_dict(baseline_show["summary"]),
        _expect_dict(candidate_show["summary"]),
    )
    baseline_sources = _sources_by_key(_expect_list(baseline_show["groups"]))
    candidate_sources = _sources_by_key(_expect_list(candidate_show["groups"]))
    changed_sources = []
    unchanged_source_count = 0
    for key in sorted(set(baseline_sources) | set(candidate_sources)):
        baseline_source = baseline_sources.get(key)
        candidate_source = candidate_sources.get(key)
        if baseline_source is None:
            changed_sources.append(
                {
                    "identity": candidate_source["identity"],
                    "status": "added",
                    "candidate": candidate_source,
                }
            )
            continue
        if candidate_source is None:
            changed_sources.append(
                {
                    "identity": baseline_source["identity"],
                    "status": "removed",
                    "baseline": baseline_source,
                }
            )
            continue

        source_diff = _diff_source(baseline_source, candidate_source)
        if source_diff is None:
            unchanged_source_count += 1
            continue
        changed_sources.append(source_diff)

    return {
        "summary": summary_diff,
        "changed_source_count": len(changed_sources),
        "unchanged_source_count": unchanged_source_count,
        "sources": changed_sources,
    }


def append_subgroup_access_show_text(
    lines: list[str],
    subgroup_access: dict[str, object],
) -> None:
    """Appends the human-readable subgroup-access view."""
    lines.extend(("", "Wave memory access (compiler analysis)"))
    _append_summary(lines, _expect_dict(subgroup_access["summary"]), indent="  ")
    lines.append(f"  groups: {subgroup_access['group_count']}")
    for group_value in _expect_list(subgroup_access["groups"]):
        _append_group(lines, _expect_dict(group_value), indent="  ")


def append_subgroup_access_diff_text(
    lines: list[str],
    subgroup_access: dict[str, object],
) -> None:
    """Appends the human-readable subgroup-access diff."""
    lines.extend(("", "Wave memory access diff (compiler analysis)"))
    availability = subgroup_access.get("availability")
    if isinstance(availability, dict):
        lines.append(
            f"  evidence: {availability['baseline']} -> {availability['candidate']}"
        )
        return

    summary = _expect_dict(subgroup_access["summary"])
    if _diff_has_changes(summary):
        lines.append("  Aggregate packets")
        _append_diff_metrics(lines, summary, indent="    ")
    else:
        lines.append("  aggregate packets: unchanged")
    lines.append(
        f"  sources: {subgroup_access['changed_source_count']} changed, "
        f"{subgroup_access['unchanged_source_count']} unchanged"
    )
    for source_value in _expect_list(subgroup_access["sources"]):
        source = _expect_dict(source_value)
        identity = _expect_dict(source["identity"])
        status = source["status"]
        lines.append(f"  {_format_source_identity(identity)}: {status}")
        if status == "added":
            _append_source(
                lines,
                _expect_dict(source["candidate"]),
                indent="    ",
                include_identity=False,
            )
            continue
        if status == "removed":
            _append_source(
                lines,
                _expect_dict(source["baseline"]),
                indent="    ",
                include_identity=False,
            )
            continue

        source_summary = _expect_dict(source["summary"])
        if _diff_has_changes(source_summary):
            lines.append("    aggregate packets")
            _append_diff_metrics(lines, source_summary, indent="      ")
        else:
            lines.append("    aggregate packets: unchanged")
        lines.append(
            f"    variants: {len(_expect_list(source['added_variants']))} added, "
            f"{len(_expect_list(source['removed_variants']))} removed, "
            f"{len(_expect_list(source['changed_variants']))} count-changed, "
            f"{source['unchanged_variant_count']} unchanged"
        )
        for variant_value in _expect_list(source["removed_variants"]):
            lines.append("    removed")
            _append_group(
                lines,
                _expect_dict(variant_value),
                indent="      ",
                include_source_identity=False,
            )
        for variant_value in _expect_list(source["added_variants"]):
            lines.append("    added")
            _append_group(
                lines,
                _expect_dict(variant_value),
                indent="      ",
                include_source_identity=False,
            )
        for variant_value in _expect_list(source["changed_variants"]):
            variant = _expect_dict(variant_value)
            identity = _expect_dict(variant["identity"])
            lines.append(f"    {_format_variant_identity(identity)}: counts changed")
            _append_diff_metrics(
                lines,
                _expect_dict(variant["summary"]),
                indent="      ",
            )


def _optional_memory(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    source_low = document.report.get("source_low")
    if source_low is None:
        return None
    source_low_object = _require_object(source_low, f"{document.source}.source_low")
    memory = source_low_object.get("memory")
    if memory is None:
        return None
    return _require_object(memory, f"{document.source}.source_low.memory")


def _show_group(value: object, source: str) -> dict[str, object]:
    group = _require_object(value, source)
    identity = {
        field: _identity_component(
            group.get(field),
            f"{source}.{field}",
            integer=field in _INTEGER_IDENTITY_FIELDS,
            required=field == "source_op_kind",
        )
        for field in _GROUP_IDENTITY_FIELDS
    }
    access = _show_access(group.get("access"), f"{source}.access")
    summary = _show_metrics(
        _require_object(group.get("summary"), f"{source}.summary"),
        f"{source}.summary",
    )
    _validate_group_summary(access, summary, f"{source}.summary")
    return {
        "report_index": _require_nonnegative_integer(
            group.get("index"), f"{source}.index"
        ),
        "identity": identity,
        "access": access,
        "summary": summary,
    }


def _show_access(value: object, source: str) -> dict[str, object]:
    access = _require_object(value, source)
    proof = _require_string(access.get("proof"), f"{source}.proof")
    if proof not in ("exact", "unknown"):
        raise CompileReportError(
            f"{source}.proof: expected 'exact' or 'unknown', got {proof!r}"
        )
    unknown_reason = access.get("unknown_reason")
    if unknown_reason is not None and not isinstance(unknown_reason, str):
        raise CompileReportError(f"{source}.unknown_reason: expected string or null")
    if proof == "exact" and unknown_reason is not None:
        raise CompileReportError(
            f"{source}.unknown_reason: exact access cannot have an unknown reason"
        )
    if proof == "unknown" and not unknown_reason:
        raise CompileReportError(
            f"{source}.unknown_reason: unknown access requires a reason"
        )

    address_value = _require_object(access.get("address"), f"{source}.address")
    address: dict[str, object] = {
        field: _require_string(address_value.get(field), f"{source}.address.{field}")
        for field in _ADDRESS_STRING_FIELDS
    }
    address.update(
        {
            field: _require_nonnegative_integer(
                address_value.get(field), f"{source}.address.{field}"
            )
            for field in _ADDRESS_INTEGER_FIELDS
        }
    )
    term_values = _require_list(
        address_value.get("lane_terms"), f"{source}.address.lane_terms"
    )
    terms = []
    for index, term_value in enumerate(term_values):
        term_source = f"{source}.address.lane_terms[{index}]"
        term = _require_object(term_value, term_source)
        divisor = _require_nonnegative_integer(
            term.get("divisor"), f"{term_source}.divisor"
        )
        if divisor == 0:
            raise CompileReportError(
                f"{term_source}.divisor: expected positive integer"
            )
        terms.append(
            {
                "divisor": divisor,
                "modulus": _require_nonnegative_integer(
                    term.get("modulus"), f"{term_source}.modulus"
                ),
                "byte_stride": _require_nonnegative_integer(
                    term.get("byte_stride"), f"{term_source}.byte_stride"
                ),
            }
        )
    address["lane_terms"] = terms
    if address["subgroup_size"] == 0:
        raise CompileReportError(f"{source}.address.subgroup_size: expected positive")

    shown: dict[str, object] = {
        "proof": proof,
        "address": address,
    }
    if unknown_reason is not None:
        shown["unknown_reason"] = unknown_reason

    geometry_value = access.get("geometry", _MISSING)
    if proof == "unknown":
        if geometry_value is not _MISSING:
            raise CompileReportError(
                f"{source}.geometry: unknown access cannot claim exact geometry"
            )
        return shown
    geometry_object = _require_object(geometry_value, f"{source}.geometry")
    coverage = _require_string(
        geometry_object.get("interval_coverage"),
        f"{source}.geometry.interval_coverage",
    )
    if coverage not in ("dense", "gapped"):
        raise CompileReportError(
            f"{source}.geometry.interval_coverage: expected 'dense' or 'gapped', "
            f"got {coverage!r}"
        )
    geometry: dict[str, object] = {"interval_coverage": coverage}
    geometry.update(
        {
            field: _require_nonnegative_integer(
                geometry_object.get(field), f"{source}.geometry.{field}"
            )
            for field in _GEOMETRY_INTEGER_FIELDS
        }
    )
    _validate_geometry(address, geometry, f"{source}.geometry")
    shown["geometry"] = geometry
    return shown


def _validate_geometry(
    address: dict[str, object],
    geometry: dict[str, object],
    source: str,
) -> None:
    subgroup_size = _expect_integer(address["subgroup_size"])
    packet_bytes = _expect_integer(address["per_lane_packet_bytes"])
    if packet_bytes == 0:
        raise CompileReportError(f"{source}: exact geometry requires packet bytes")
    requested = _expect_integer(geometry["subgroup_requested_bytes"])
    unique = _expect_integer(geometry["subgroup_unique_bytes"])
    span = _expect_integer(geometry["subgroup_span_bytes"])
    expected_requested = subgroup_size * packet_bytes
    if requested != expected_requested:
        raise CompileReportError(
            f"{source}.subgroup_requested_bytes: expected {expected_requested}, "
            f"got {requested}"
        )
    if not 0 < unique <= requested:
        raise CompileReportError(
            f"{source}.subgroup_unique_bytes: expected 1..{requested}, got {unique}"
        )
    if span < unique:
        raise CompileReportError(
            f"{source}.subgroup_span_bytes: expected at least {unique}, got {span}"
        )
    distinct = _expect_integer(geometry["distinct_lane_address_count"])
    if not 0 < distinct <= subgroup_size:
        raise CompileReportError(
            f"{source}.distinct_lane_address_count: expected 1..{subgroup_size}, "
            f"got {distinct}"
        )
    regions = _expect_integer(geometry["contiguous_region_count"])
    if not 0 < regions <= distinct:
        raise CompileReportError(
            f"{source}.contiguous_region_count: expected 1..{distinct}, got {regions}"
        )
    maximum_gap = _expect_integer(geometry["maximum_uncovered_gap_bytes"])
    if geometry["interval_coverage"] == "dense":
        if span != unique or regions != 1 or maximum_gap != 0:
            raise CompileReportError(
                f"{source}: dense coverage requires span=unique, one region, "
                "and zero uncovered gap"
            )
    elif span == unique or regions == 1 or maximum_gap == 0:
        raise CompileReportError(
            f"{source}: gapped coverage requires span>unique, multiple regions, "
            "and a nonzero uncovered gap"
        )


def _validate_group_target_subgroup_size(
    document: CompileReportDocument,
    group: dict[str, object],
    source: str,
) -> None:
    identity = _expect_dict(group["identity"])
    function = identity["function"]
    matching_entries = []
    if function is None:
        if len(document.entries) == 1:
            matching_entries.append(document.entries[0])
    else:
        matching_entries = [
            entry for entry in document.entries if entry.get("function") == function
        ]
    if function is not None and not matching_entries:
        for entry in document.entries:
            if function in (
                entry.get("source_function"),
                entry.get("target_export"),
                entry.get("target_export_symbol"),
            ):
                matching_entries.append(entry)
    if len(matching_entries) > 1:
        raise CompileReportError(
            f"{source}.function: {function!r} matches multiple report entries"
        )
    entry = matching_entries[0] if matching_entries else None
    if entry is not None and "target_resources" in entry:
        target_resources_value = entry["target_resources"]
        target_resources_source = (
            f"{document.source}.entries.rows[{entry['index']}].target_resources"
        )
    else:
        target_resources_value = document.report.get("target_resources")
        target_resources_source = f"{document.source}.target_resources"
    if target_resources_value is None:
        return
    target_resources = _require_object(
        target_resources_value,
        target_resources_source,
    )
    target_subgroup_size_value = target_resources.get("subgroup_size")
    if target_subgroup_size_value is None:
        return
    target_subgroup_size = _require_nonnegative_integer(
        target_subgroup_size_value,
        f"{target_resources_source}.subgroup_size",
    )
    if target_subgroup_size == 0:
        return
    access = _expect_dict(group["access"])
    address = _expect_dict(access["address"])
    subgroup_size = _expect_integer(address["subgroup_size"])
    if subgroup_size != target_subgroup_size:
        raise CompileReportError(
            f"{source}.access.address.subgroup_size: expected target subgroup "
            f"size {target_subgroup_size}, got {subgroup_size}"
        )


def _validate_group_summary(
    access: dict[str, object],
    summary: dict[str, object],
    source: str,
) -> None:
    modeled = _expect_integer(summary["modeled_packet_count"])
    exact = _expect_integer(summary["exact_packet_count"])
    unknown = _expect_integer(summary["unknown_packet_count"])
    if modeled != exact + unknown:
        raise CompileReportError(
            f"{source}: modeled packets must equal exact plus unknown packets"
        )
    dense = _expect_integer(summary["dense_packet_count"])
    gapped = _expect_integer(summary["gapped_packet_count"])
    overlapping = _expect_integer(summary["overlapping_packet_count"])
    if dense + gapped != exact or overlapping > exact:
        raise CompileReportError(
            f"{source}: structural shape counts are inconsistent with exact packets"
        )
    dynamic_exact = _expect_integer(summary["exact_dynamic_packet_count"])
    dynamic_unknown = _expect_integer(summary["unknown_dynamic_packet_count"])
    if dynamic_exact + dynamic_unknown != modeled:
        raise CompileReportError(
            f"{source}: dynamic proof counts must partition modeled packets"
        )
    dynamic_packets = _expect_integer(summary["dynamic_packet_count"])
    dynamic_dense = _expect_integer(summary["dynamic_dense_packet_count"])
    dynamic_gapped = _expect_integer(summary["dynamic_gapped_packet_count"])
    dynamic_overlapping = _expect_integer(summary["dynamic_overlapping_packet_count"])
    if dynamic_dense + dynamic_gapped != dynamic_packets:
        raise CompileReportError(
            f"{source}: dynamic shape counts must partition packet executions"
        )
    if dynamic_overlapping > dynamic_packets:
        raise CompileReportError(
            f"{source}: dynamic overlap count exceeds packet executions"
        )

    if access["proof"] == "unknown":
        if (
            exact != 0
            or unknown != modeled
            or dynamic_exact != 0
            or dynamic_unknown != modeled
            or dynamic_packets != 0
        ):
            raise CompileReportError(
                f"{source}: unknown access group contains exact geometry counts"
            )
        return
    if unknown != 0 or exact != modeled:
        raise CompileReportError(
            f"{source}: exact access group contains unknown geometry counts"
        )
    geometry = _expect_dict(access["geometry"])
    expected_dense = exact if geometry["interval_coverage"] == "dense" else 0
    expected_gapped = exact if geometry["interval_coverage"] == "gapped" else 0
    expected_overlap = (
        exact
        if geometry["subgroup_requested_bytes"] > geometry["subgroup_unique_bytes"]
        else 0
    )
    if (dense, gapped, overlapping) != (
        expected_dense,
        expected_gapped,
        expected_overlap,
    ):
        raise CompileReportError(
            f"{source}: group summary disagrees with retained exact geometry"
        )
    if dynamic_packets != 0:
        expected_dynamic_dense = (
            dynamic_packets if geometry["interval_coverage"] == "dense" else 0
        )
        expected_dynamic_gapped = (
            dynamic_packets if geometry["interval_coverage"] == "gapped" else 0
        )
        expected_dynamic_overlap = dynamic_packets if expected_overlap != 0 else 0
        if (dynamic_dense, dynamic_gapped, dynamic_overlapping) != (
            expected_dynamic_dense,
            expected_dynamic_gapped,
            expected_dynamic_overlap,
        ):
            raise CompileReportError(
                f"{source}: dynamic summary disagrees with retained exact geometry"
            )


def _show_metrics(summary: dict[str, object], source: str) -> dict[str, object]:
    metrics = {}
    for spec in _METRIC_SPECS:
        value = _lookup(summary, spec.path)
        metrics[spec.key] = _require_nonnegative_integer(value, f"{source}.{spec.path}")
    return metrics


def _validate_unique_variants(groups: list[dict[str, object]], source: str) -> None:
    variants = set()
    for position, group in enumerate(groups):
        key = _variant_key(group)
        if key in variants:
            raise CompileReportError(
                f"{source}.source_low.memory.subgroup_access_groups[{position}]: "
                "duplicate semantic access variant"
            )
        variants.add(key)


def _sum_group_summaries(groups: list[dict[str, object]]) -> dict[str, object]:
    result = {spec.key: 0 for spec in _METRIC_SPECS}
    for group in groups:
        summary = _expect_dict(group["summary"])
        for spec in _METRIC_SPECS:
            result[spec.key] += _expect_integer(summary[spec.key])
    return result


def _sources_by_key(
    values: list[object],
) -> dict[tuple[tuple[int, object], ...], dict[str, object]]:
    source_groups: dict[tuple[tuple[int, object], ...], list[dict[str, object]]] = {}
    for value in values:
        group = _expect_dict(value)
        key = _source_key(group)
        source_groups.setdefault(key, []).append(group)
    sources = {}
    for key, groups in source_groups.items():
        groups.sort(key=_group_sort_key)
        identity = _expect_dict(groups[0]["identity"])
        sources[key] = {
            "identity": {field: identity[field] for field in _SOURCE_IDENTITY_FIELDS},
            "summary": _sum_group_summaries(groups),
            "variant_count": len(groups),
            "variants": groups,
        }
    return sources


def _diff_source(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object] | None:
    baseline_variants = {
        _variant_key(_expect_dict(value)): _expect_dict(value)
        for value in _expect_list(baseline["variants"])
    }
    candidate_variants = {
        _variant_key(_expect_dict(value)): _expect_dict(value)
        for value in _expect_list(candidate["variants"])
    }
    added = []
    removed = []
    changed = []
    unchanged_variant_count = 0
    for key in sorted(set(baseline_variants) | set(candidate_variants)):
        baseline_variant = baseline_variants.get(key)
        candidate_variant = candidate_variants.get(key)
        if baseline_variant is None:
            added.append(candidate_variant)
            continue
        if candidate_variant is None:
            removed.append(baseline_variant)
            continue
        summary = _diff_metrics(
            _expect_dict(baseline_variant["summary"]),
            _expect_dict(candidate_variant["summary"]),
        )
        if not _diff_has_changes(summary):
            unchanged_variant_count += 1
            continue
        changed.append(
            {
                "identity": baseline_variant["identity"],
                "access": baseline_variant["access"],
                "summary": summary,
            }
        )
    if not added and not removed and not changed:
        return None
    return {
        "identity": baseline["identity"],
        "status": "changed",
        "summary": _diff_metrics(
            _expect_dict(baseline["summary"]),
            _expect_dict(candidate["summary"]),
        ),
        "added_variants": added,
        "removed_variants": removed,
        "changed_variants": changed,
        "unchanged_variant_count": unchanged_variant_count,
    }


def _diff_metrics(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    changed: dict[str, object] = {}
    unchanged_count = 0
    for spec in _METRIC_SPECS:
        baseline_value = _expect_integer(baseline[spec.key])
        candidate_value = _expect_integer(candidate[spec.key])
        if baseline_value == candidate_value:
            unchanged_count += 1
            continue
        delta = candidate_value - baseline_value
        metric: dict[str, object] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
        if baseline_value != 0:
            metric["change_percent"] = delta * 100.0 / baseline_value
        changed[spec.key] = metric
    return {
        "changed": changed,
        "incomplete": {},
        "unchanged_count": unchanged_count,
        "unavailable_count": 0,
    }


def _diff_has_changes(group: dict[str, object]) -> bool:
    return bool(group["changed"] or group["incomplete"])


def _append_summary(
    lines: list[str], summary: dict[str, object], *, indent: str
) -> None:
    lines.append(
        f"{indent}proof: {summary['exact_packet_count']} exact, "
        f"{summary['unknown_packet_count']} unknown of "
        f"{summary['modeled_packet_count']} modeled packets"
    )
    lines.append(
        f"{indent}shapes: {summary['dense_packet_count']} dense, "
        f"{summary['gapped_packet_count']} gapped, "
        f"{summary['overlapping_packet_count']} overlapping"
    )
    lines.append(
        f"{indent}dynamic: {summary['dynamic_packet_count']} packet executions "
        f"from {summary['exact_dynamic_packet_count']} source packets with exact "
        "geometry/count contributions and "
        f"{summary['unknown_dynamic_packet_count']} without; "
        f"{summary['dynamic_dense_packet_count']} dense, "
        f"{summary['dynamic_gapped_packet_count']} gapped, "
        f"{summary['dynamic_overlapping_packet_count']} overlapping"
    )


def _append_diff_metrics(
    lines: list[str], metric_values: dict[str, object], *, indent: str
) -> None:
    changed = _expect_dict(metric_values["changed"])
    for spec in _METRIC_SPECS:
        metric_value = changed.get(spec.key)
        if metric_value is None:
            continue
        metric = _expect_dict(metric_value)
        suffix = f", delta {_format_signed_integer(metric['delta'])}"
        if "change_percent" in metric:
            suffix += f" ({metric['change_percent']:+.2f}%)"
        lines.append(
            f"{indent}{spec.label}: {_format_integer(metric['baseline'])} -> "
            f"{_format_integer(metric['candidate'])}{suffix}"
        )


def _append_source(
    lines: list[str],
    source: dict[str, object],
    *,
    indent: str,
    include_identity: bool = True,
) -> None:
    if include_identity:
        lines.append(
            f"{indent}{_format_source_identity(_expect_dict(source['identity']))}"
        )
        indent += "  "
    _append_summary(lines, _expect_dict(source["summary"]), indent=indent)
    lines.append(f"{indent}variants: {source['variant_count']}")
    for group_value in _expect_list(source["variants"]):
        _append_group(
            lines,
            _expect_dict(group_value),
            indent=indent + "  ",
            include_source_identity=False,
        )


def _append_group(
    lines: list[str],
    group: dict[str, object],
    *,
    indent: str,
    include_source_identity: bool = True,
) -> None:
    identity = _expect_dict(group["identity"])
    if include_source_identity:
        lines.append(f"{indent}{_format_group_identity(identity)}")
    else:
        lines.append(f"{indent}{_format_variant_identity(identity)}")
    access = _expect_dict(group["access"])
    address = _expect_dict(access["address"])
    proof_line = f"{indent}  proof: {access['proof']}"
    if access["proof"] == "unknown":
        proof_line += f" ({access['unknown_reason']})"
    proof_line += (
        f"; lane address {address['lane_address_proof']}; "
        f"active lanes {address['active_lane_proof']}"
    )
    lines.append(proof_line)
    packet_bytes = _expect_integer(address["per_lane_packet_bytes"])
    packet = f"{packet_bytes} B/lane" if packet_bytes else "packet width unknown"
    lines.append(
        f"{indent}  address: {address['subgroup_size']} lanes, {packet}; "
        f"{_format_lane_mapping(address)}"
    )
    geometry = access.get("geometry")
    if not isinstance(geometry, dict):
        return
    overlap = geometry["subgroup_requested_bytes"] > geometry["subgroup_unique_bytes"]
    coverage = str(geometry["interval_coverage"])
    if overlap:
        coverage += ", overlapping"
    lines.append(
        f"{indent}  geometry: {coverage}; requested "
        f"{_format_bytes(geometry['subgroup_requested_bytes'])}, unique "
        f"{_format_bytes(geometry['subgroup_unique_bytes'])}, span "
        f"{_format_bytes(geometry['subgroup_span_bytes'])}"
    )
    lines.append(
        f"{indent}  distribution: {geometry['distinct_lane_address_count']} "
        f"distinct {_plural('start', geometry['distinct_lane_address_count'])}, "
        f"{geometry['contiguous_region_count']} "
        f"{_plural('region', geometry['contiguous_region_count'])}, "
        f"max gap {_format_bytes(geometry['maximum_uncovered_gap_bytes'])}, "
        "max adjacent-lane delta "
        f"{_format_bytes(geometry['maximum_adjacent_lane_delta_bytes'])}"
    )


def _format_lane_mapping(address: dict[str, object]) -> str:
    terms = [_expect_dict(value) for value in _expect_list(address["lane_terms"])]
    if not terms:
        return f"{address['lane_mapping']} lane offset"
    expressions = []
    for term in terms:
        divisor = _expect_integer(term["divisor"])
        modulus = _expect_integer(term["modulus"])
        stride = _expect_integer(term["byte_stride"])
        if divisor == 1:
            lane_digit = "lane"
        else:
            lane_digit = f"floor(lane/{divisor})"
        if modulus != 0:
            lane_digit = f"({lane_digit} % {modulus})"
        expressions.append(f"{lane_digit} * {stride} B")
    return f"{address['lane_mapping']} offset: " + " + ".join(expressions)


def _format_source_identity(identity: dict[str, object]) -> str:
    function = identity.get("function") or "<unnamed>"
    source_op = identity.get("source_op") or "<unknown-op>"
    source_root = identity.get("source_root")
    if not source_root and identity.get("source_root_argument_index") is not None:
        source_root = f"arg{identity['source_root_argument_index']}"
    memory_space = identity.get("memory_space") or "unknown-space"
    operation = identity.get("operation") or "unknown-operation"
    return (
        f"{function}: {source_op} {source_root or '<unknown-root>'} "
        f"({memory_space} {operation})"
    )


def _format_group_identity(identity: dict[str, object]) -> str:
    return f"{_format_source_identity(identity)} {_format_variant_identity(identity)}"


def _format_variant_identity(identity: dict[str, object]) -> str:
    packet = identity.get("packet") or "<unknown-packet>"
    strategy = identity.get("strategy")
    suffix = f"; strategy={strategy}" if strategy else ""
    return f"[{packet}{suffix}]"


def _group_sort_key(group: dict[str, object]) -> tuple[object, ...]:
    return (_source_key(group), _variant_key(group))


def _source_key(group: dict[str, object]) -> tuple[tuple[int, object], ...]:
    identity = _expect_dict(group["identity"])
    return tuple(
        _sortable_component(identity[field]) for field in _SOURCE_IDENTITY_FIELDS
    )


def _variant_key(group: dict[str, object]) -> tuple[object, ...]:
    identity = _expect_dict(group["identity"])
    return (
        _source_key(group),
        _sortable_component(identity["packet"]),
        _sortable_component(identity["strategy"]),
        _freeze(group["access"]),
    )


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _sortable_component(value: object) -> tuple[int, object]:
    if value is None:
        return (0, "")
    if isinstance(value, int):
        return (1, value)
    return (2, str(value))


def _identity_component(
    value: object, source: str, *, integer: bool, required: bool
) -> object:
    if value is None:
        if required:
            expected = "integer" if integer else "string"
            raise CompileReportError(f"{source}: expected {expected}")
        return value
    if integer and isinstance(value, int) and not isinstance(value, bool):
        return value
    if not integer and isinstance(value, str):
        return value
    expected = "integer or null" if integer else "string or null"
    raise CompileReportError(f"{source}: expected {expected}")


def _lookup(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{source}: expected object")
    return value


def _require_list(value: object, source: str) -> list[object]:
    if not isinstance(value, list):
        raise CompileReportError(f"{source}: expected array")
    return value


def _require_string(value: object, source: str) -> str:
    if not isinstance(value, str):
        raise CompileReportError(f"{source}: expected string")
    return value


def _require_integer(value: object, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CompileReportError(f"{source}: expected integer")
    return value


def _require_nonnegative_integer(value: object, source: str) -> int:
    result = _require_integer(value, source)
    if result < 0:
        raise CompileReportError(f"{source}: expected nonnegative integer")
    return result


def _expect_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("invalid internal subgroup-access report object")
    return value


def _expect_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("invalid internal subgroup-access report array")
    return value


def _expect_integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("invalid internal subgroup-access report integer")
    return value


def _format_integer(value: object) -> str:
    return f"{_expect_integer(value):,}"


def _format_signed_integer(value: object) -> str:
    return f"{_expect_integer(value):+,}"


def _format_bytes(value: object) -> str:
    return f"{_expect_integer(value):,} B"


def _plural(noun: str, count: object) -> str:
    return noun if _expect_integer(count) == 1 else f"{noun}s"
