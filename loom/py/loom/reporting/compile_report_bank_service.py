# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bank-service views, semantic diffs, and text formatting."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportDocument, CompileReportError

_MISSING = object()


@dataclass(frozen=True)
class _MetricSpec:
    """One stable bank-service scalar."""

    key: str
    label: str
    path: str


_METRIC_SPECS = (
    _MetricSpec("modeled_packet_count", "modeled packets", "modeled_packet_count"),
    _MetricSpec("exact_packet_count", "exact packets", "exact_packet_count"),
    _MetricSpec("unknown_packet_count", "unknown packets", "unknown_packet_count"),
    _MetricSpec(
        "conflict_free_packet_count",
        "conflict-free packets",
        "structural.conflict_free_packet_count",
    ),
    _MetricSpec(
        "conflicted_packet_count",
        "conflicted packets",
        "structural.conflicted_packet_count",
    ),
    _MetricSpec(
        "required_round_count",
        "structural required rounds",
        "structural.required_round_count",
    ),
    _MetricSpec(
        "uncontended_round_count",
        "structural uncontended rounds",
        "structural.uncontended_round_count",
    ),
    _MetricSpec(
        "extra_round_count",
        "structural extra rounds",
        "structural.extra_round_count",
    ),
    _MetricSpec(
        "maximum_request_multiplicity",
        "maximum request multiplicity",
        "structural.maximum_request_multiplicity",
    ),
    _MetricSpec(
        "exact_dynamic_packet_count",
        "dynamically exact source packets",
        "dynamic.exact_packet_count",
    ),
    _MetricSpec(
        "unknown_dynamic_packet_count",
        "dynamically unknown source packets",
        "dynamic.unknown_packet_count",
    ),
    _MetricSpec(
        "dynamic_packet_count",
        "dynamic packet executions",
        "dynamic.packet_count",
    ),
    _MetricSpec(
        "dynamic_required_round_count",
        "dynamic required rounds",
        "dynamic.required_round_count",
    ),
    _MetricSpec(
        "dynamic_uncontended_round_count",
        "dynamic uncontended rounds",
        "dynamic.uncontended_round_count",
    ),
    _MetricSpec(
        "dynamic_extra_round_count",
        "dynamic extra rounds",
        "dynamic.extra_round_count",
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

_MODEL_FIELDS = (
    "key",
    "revision",
    "evidence",
    "request_policy",
    "wave_size",
    "bank_count",
    "bank_word_bytes",
    "packet_bank_words",
)


def build_bank_service_show(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds the compact bank-service view when evidence is available."""
    memory = _optional_memory(document)
    if memory is None:
        return None
    summary = memory.get("bank_service")
    groups = memory.get("bank_service_groups")
    if summary is None and groups is None:
        return None
    summary_object = _require_object(
        summary,
        f"{document.source}.source_low.memory.bank_service",
    )
    group_values = _require_list(
        groups,
        f"{document.source}.source_low.memory.bank_service_groups",
    )
    expected_group_count = memory.get("bank_service_group_count")
    if (
        not isinstance(expected_group_count, int)
        or isinstance(expected_group_count, bool)
        or expected_group_count != len(group_values)
    ):
        raise CompileReportError(
            f"{document.source}.source_low.memory.bank_service_group_count: "
            f"expected {len(group_values)}, got {expected_group_count!r}"
        )
    shown_groups = [
        _show_group(
            value,
            f"{document.source}.source_low.memory.bank_service_groups[{index}]",
        )
        for index, value in enumerate(group_values)
    ]
    shown_groups.sort(key=_group_key)
    return {
        "summary": _show_metrics(
            summary_object,
            f"{document.source}.source_low.memory.bank_service",
        ),
        "group_count": len(shown_groups),
        "groups": shown_groups,
    }


def build_bank_service_diff(
    baseline: CompileReportDocument,
    candidate: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds a semantic bank-service diff without globally netting groups."""
    baseline_show = build_bank_service_show(baseline)
    candidate_show = build_bank_service_show(candidate)
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
            "changed_group_count": 0,
            "unchanged_group_count": 0,
            "groups": [],
        }

    summary_diff = _diff_metrics(
        _expect_dict(baseline_show["summary"]),
        _expect_dict(candidate_show["summary"]),
    )
    baseline_groups = _groups_by_key(
        _expect_list(baseline_show["groups"]),
        baseline.source,
    )
    candidate_groups = _groups_by_key(
        _expect_list(candidate_show["groups"]),
        candidate.source,
    )
    changed_groups = []
    unchanged_group_count = 0
    for key in sorted(set(baseline_groups) | set(candidate_groups)):
        baseline_group = baseline_groups.get(key)
        candidate_group = candidate_groups.get(key)
        if baseline_group is None:
            changed_groups.append(
                {
                    "identity": candidate_group["identity"],
                    "status": "added",
                    "candidate": candidate_group,
                }
            )
            continue
        if candidate_group is None:
            changed_groups.append(
                {
                    "identity": baseline_group["identity"],
                    "status": "removed",
                    "baseline": baseline_group,
                }
            )
            continue

        model_diff = _diff_model(
            _expect_dict(baseline_group["model"]),
            _expect_dict(candidate_group["model"]),
        )
        baseline_summary = _expect_dict(baseline_group["summary"])
        candidate_summary = _expect_dict(candidate_group["summary"])
        service_diff = _diff_metrics(baseline_summary, candidate_summary)
        baseline_unknown = baseline_group.get("unknown_evidence")
        candidate_unknown = candidate_group.get("unknown_evidence")
        if (
            not model_diff
            and not _diff_has_changes(service_diff)
            and baseline_unknown == candidate_unknown
        ):
            unchanged_group_count += 1
            continue
        group_diff: dict[str, object] = {
            "identity": baseline_group["identity"],
            "status": "changed",
            "proof_loss": _proof_lost(baseline_summary, candidate_summary),
            "service": service_diff,
        }
        if model_diff:
            group_diff["model"] = model_diff
        if baseline_unknown != candidate_unknown:
            group_diff["unknown_evidence"] = {
                "baseline": baseline_unknown,
                "candidate": candidate_unknown,
            }
        changed_groups.append(group_diff)

    return {
        "summary": summary_diff,
        "changed_group_count": len(changed_groups),
        "unchanged_group_count": unchanged_group_count,
        "groups": changed_groups,
    }


def append_bank_service_show_text(
    lines: list[str],
    bank_service: dict[str, object],
) -> None:
    """Appends the human-readable bank-service view."""
    lines.extend(("", "Bank service (compiler analysis)"))
    _append_metrics(lines, _expect_dict(bank_service["summary"]), indent="  ")
    actionable_groups = [
        _expect_dict(value)
        for value in _expect_list(bank_service["groups"])
        if _group_is_actionable(_expect_dict(value))
    ]
    lines.append(
        f"  groups: {bank_service['group_count']} total, "
        f"{len(actionable_groups)} conflicted or incomplete"
    )
    for group in actionable_groups:
        _append_group(lines, group, indent="  ")


def append_bank_service_diff_text(
    lines: list[str],
    bank_service: dict[str, object],
) -> None:
    """Appends the human-readable semantic bank-service diff."""
    lines.extend(("", "Bank service diff (compiler analysis)"))
    availability = bank_service.get("availability")
    if isinstance(availability, dict):
        lines.append(
            f"  evidence: {availability['baseline']} -> {availability['candidate']}"
        )
        return
    summary = _expect_dict(bank_service["summary"])
    if _diff_has_changes(summary):
        lines.append("  Aggregate service")
        _append_diff_metrics(lines, summary, indent="    ")
    else:
        lines.append("  aggregate service: unchanged")
    lines.append(
        f"  groups: {bank_service['changed_group_count']} changed, "
        f"{bank_service['unchanged_group_count']} unchanged"
    )
    for group_value in _expect_list(bank_service["groups"]):
        group = _expect_dict(group_value)
        status = group["status"]
        identity = _expect_dict(group["identity"])
        lines.append(f"  {_format_group_identity(identity)}: {status}")
        if status == "added":
            _append_group(
                lines,
                _expect_dict(group["candidate"]),
                indent="    ",
                include_identity=False,
            )
            continue
        if status == "removed":
            _append_group(
                lines,
                _expect_dict(group["baseline"]),
                indent="    ",
                include_identity=False,
            )
            continue
        if group.get("proof_loss") is True:
            lines.append("    exact proof coverage regressed")
        model = group.get("model")
        if isinstance(model, dict):
            for field, change_value in model.items():
                change = _expect_dict(change_value)
                lines.append(
                    f"    model {field}: {change['baseline']} -> {change['candidate']}"
                )
        _append_diff_metrics(
            lines,
            _expect_dict(group["service"]),
            indent="    ",
        )
        unknown_evidence = group.get("unknown_evidence")
        if isinstance(unknown_evidence, dict):
            lines.append(
                "    unknown evidence: "
                f"{unknown_evidence['baseline']} -> "
                f"{unknown_evidence['candidate']}"
            )


def _optional_memory(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    source_low = document.report.get("source_low")
    if source_low is None:
        return None
    source_low_object = _require_object(
        source_low,
        f"{document.source}.source_low",
    )
    memory = source_low_object.get("memory")
    if memory is None:
        return None
    return _require_object(memory, f"{document.source}.source_low.memory")


def _show_group(value: object, source: str) -> dict[str, object]:
    group = _require_object(value, source)
    identity = {
        field: _identity_component(group.get(field), f"{source}.{field}")
        for field in _GROUP_IDENTITY_FIELDS
    }
    model = _require_object(group.get("model"), f"{source}.model")
    summary = _require_object(group.get("summary"), f"{source}.summary")
    shown: dict[str, object] = {
        "report_index": _require_integer(group.get("index"), f"{source}.index"),
        "identity": identity,
        "model": {
            field: _model_component(model.get(field), f"{source}.model.{field}")
            for field in _MODEL_FIELDS
        },
        "summary": _show_metrics(summary, f"{source}.summary"),
    }
    if "unknown_evidence" in group:
        unknown_evidence = _require_object(
            group["unknown_evidence"],
            f"{source}.unknown_evidence",
        )
        reason = unknown_evidence.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise CompileReportError(
                f"{source}.unknown_evidence.reason: expected string or null"
            )
        mixed_reasons = unknown_evidence.get("mixed_reasons")
        if not isinstance(mixed_reasons, bool):
            raise CompileReportError(
                f"{source}.unknown_evidence.mixed_reasons: expected boolean"
            )
        shown["unknown_evidence"] = {
            "reason": reason,
            "mixed_reasons": mixed_reasons,
        }
    return shown


def _show_metrics(
    summary: dict[str, object],
    source: str,
) -> dict[str, object]:
    metrics = {}
    for spec in _METRIC_SPECS:
        value = _lookup(summary, spec.path)
        if not isinstance(value, int) or isinstance(value, bool):
            raise CompileReportError(f"{source}.{spec.path}: expected integer")
        metrics[spec.key] = value
    return metrics


def _identity_component(value: object, source: str) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CompileReportError(f"{source}: expected string, integer, or null")


def _model_component(value: object, source: str) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CompileReportError(f"{source}: expected string or integer")


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{source}: expected object")
    return value


def _require_list(value: object, source: str) -> list[object]:
    if not isinstance(value, list):
        raise CompileReportError(f"{source}: expected array")
    return value


def _require_integer(value: object, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CompileReportError(f"{source}: expected integer")
    return value


def _group_key(
    group: dict[str, object],
) -> tuple[tuple[int, object], ...]:
    identity = _expect_dict(group["identity"])
    return tuple(
        _sortable_component(identity[field]) for field in _GROUP_IDENTITY_FIELDS
    )


def _groups_by_key(
    values: list[object],
    source: str,
) -> dict[tuple[tuple[int, object], ...], dict[str, object]]:
    groups = {}
    for position, value in enumerate(values):
        group = _expect_dict(value)
        key = _group_key(group)
        if key in groups:
            raise CompileReportError(
                f"{source}.source_low.memory.bank_service_groups[{position}]: "
                "duplicate semantic group identity"
            )
        groups[key] = group
    return groups


def _sortable_component(value: object) -> tuple[int, object]:
    if value is None:
        return (0, "")
    if isinstance(value, int):
        return (1, value)
    return (2, str(value))


def _diff_model(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        field: {
            "baseline": baseline.get(field),
            "candidate": candidate.get(field),
        }
        for field in _MODEL_FIELDS
        if baseline.get(field) != candidate.get(field)
    }


def _proof_lost(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> bool:
    return (
        candidate["exact_packet_count"] < baseline["exact_packet_count"]
        or candidate["unknown_packet_count"] > baseline["unknown_packet_count"]
    )


def _diff_metrics(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    changed: dict[str, object] = {}
    unchanged_count = 0
    for spec in _METRIC_SPECS:
        baseline_value = baseline[spec.key]
        candidate_value = candidate[spec.key]
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


def _append_metrics(
    lines: list[str],
    metric_values: dict[str, object],
    *,
    indent: str,
) -> None:
    lines.extend(
        f"{indent}{spec.label}: {_format_value(metric_values[spec.key])}"
        for spec in _METRIC_SPECS
    )


def _append_diff_metrics(
    lines: list[str],
    metric_values: dict[str, object],
    *,
    indent: str,
) -> None:
    changed = _expect_dict(metric_values["changed"])
    for spec in _METRIC_SPECS:
        metric_value = changed.get(spec.key)
        if metric_value is None:
            continue
        metric = _expect_dict(metric_value)
        suffix = f", delta {_format_signed_value(metric['delta'])}"
        if "change_percent" in metric:
            suffix += f" ({metric['change_percent']:+.2f}%)"
        lines.append(
            f"{indent}{spec.label}: "
            f"{_format_value(metric['baseline'])} -> "
            f"{_format_value(metric['candidate'])}{suffix}"
        )


def _append_group(
    lines: list[str],
    group: dict[str, object],
    *,
    indent: str,
    include_identity: bool = True,
) -> None:
    identity = _expect_dict(group["identity"])
    model = _expect_dict(group["model"])
    summary = _expect_dict(group["summary"])
    if include_identity:
        lines.append(f"{indent}{_format_group_identity(identity)}")
        indent += "  "
    lines.append(
        f"{indent}model: {model['key']} @ {model['revision']} "
        f"({model['evidence']}, {model['request_policy']})"
    )
    lines.append(
        f"{indent}proof: {summary['exact_packet_count']} exact, "
        f"{summary['unknown_packet_count']} unknown"
    )
    lines.append(
        f"{indent}structural service: "
        f"{summary['required_round_count']} required, "
        f"{summary['uncontended_round_count']} uncontended, "
        f"{summary['extra_round_count']} extra rounds"
    )
    unknown_evidence = group.get("unknown_evidence")
    if isinstance(unknown_evidence, dict):
        lines.append(
            f"{indent}unknown evidence: {unknown_evidence.get('reason') or 'mixed'}"
        )


def _group_is_actionable(group: dict[str, object]) -> bool:
    summary = _expect_dict(group["summary"])
    return summary["conflicted_packet_count"] > 0 or summary["unknown_packet_count"] > 0


def _format_group_identity(identity: dict[str, object]) -> str:
    function = identity.get("function") or "<unnamed>"
    source_op = identity.get("source_op") or "<unknown-op>"
    source_root = identity.get("source_root")
    if not source_root and identity.get("source_root_argument_index") is not None:
        source_root = f"arg{identity['source_root_argument_index']}"
    packet = identity.get("packet") or "<unknown-packet>"
    return f"{function}: {source_op} {source_root or '<unknown-root>'} [{packet}]"


def _lookup(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _format_value(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_signed_value(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:+,}"
    if isinstance(value, float):
        return f"{value:+.4g}"
    return str(value)


def _expect_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("invalid internal bank-service report object")
    return value


def _expect_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("invalid internal bank-service report array")
    return value
