# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Move-cause views, semantic diffs, and text formatting."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportError


@dataclass(frozen=True)
class CompileReportMoveCause:
    """One compiler-classified source of emitted register moves."""

    cause: str
    packet_count: int
    unit_count: int
    position: int

    def to_json_object(self) -> dict[str, object]:
        """Returns the stable public fields of the cause row."""
        return {
            "cause": self.cause,
            "packet_count": self.packet_count,
            "unit_count": self.unit_count,
        }


@dataclass(frozen=True)
class CompileReportMoveCauseSummary:
    """Validated aggregate and optional detailed move-cause evidence."""

    kind_count: int
    packet_count: int
    unit_count: int
    causes: tuple[CompileReportMoveCause, ...] | None


def parse_compile_report_move_causes(
    entry: dict[str, object],
    source: str,
) -> CompileReportMoveCauseSummary | None:
    """Parses and validates one entry's optional move-cause evidence."""
    value = entry.get("move_causes")
    if value is None:
        return None
    move_causes = _require_object(value, f"{source}.move_causes")
    kind_count = _require_count(
        move_causes.get("kind_count"), f"{source}.move_causes.kind_count"
    )
    packet_count = _require_count(
        move_causes.get("packet_count"), f"{source}.move_causes.packet_count"
    )
    unit_count = _require_count(
        move_causes.get("unit_count"), f"{source}.move_causes.unit_count"
    )
    causes_value = move_causes.get("causes")
    if causes_value is None:
        return CompileReportMoveCauseSummary(
            kind_count=kind_count,
            packet_count=packet_count,
            unit_count=unit_count,
            causes=None,
        )
    if not isinstance(causes_value, list):
        raise CompileReportError(f"{source}.move_causes.causes: expected array")

    causes = []
    seen_causes = set()
    for position, cause_value in enumerate(causes_value):
        cause_source = f"{source}.move_causes.causes[{position}]"
        cause_object = _require_object(cause_value, cause_source)
        cause = _require_string(cause_object.get("cause"), f"{cause_source}.cause")
        if cause in seen_causes:
            raise CompileReportError(
                f"{cause_source}.cause: duplicate move cause {cause!r}"
            )
        seen_causes.add(cause)
        causes.append(
            CompileReportMoveCause(
                cause=cause,
                packet_count=_require_count(
                    cause_object.get("packet_count"),
                    f"{cause_source}.packet_count",
                ),
                unit_count=_require_count(
                    cause_object.get("unit_count"),
                    f"{cause_source}.unit_count",
                ),
                position=position,
            )
        )

    if kind_count != len(causes):
        raise CompileReportError(
            f"{source}.move_causes.kind_count: expected {len(causes)}, got {kind_count}"
        )
    cause_packet_count = sum(cause.packet_count for cause in causes)
    if packet_count != cause_packet_count:
        raise CompileReportError(
            f"{source}.move_causes.packet_count: detailed causes total "
            f"{cause_packet_count}, got {packet_count}"
        )
    cause_unit_count = sum(cause.unit_count for cause in causes)
    if unit_count != cause_unit_count:
        raise CompileReportError(
            f"{source}.move_causes.unit_count: detailed causes total "
            f"{cause_unit_count}, got {unit_count}"
        )
    return CompileReportMoveCauseSummary(
        kind_count=kind_count,
        packet_count=packet_count,
        unit_count=unit_count,
        causes=tuple(causes),
    )


def build_move_cause_show(
    entry: dict[str, object],
    source: str,
) -> dict[str, object] | None:
    """Builds the compact move-cause view when evidence is available."""
    summary = parse_compile_report_move_causes(entry, source)
    if summary is None:
        return None
    view: dict[str, object] = {
        "kind_count": summary.kind_count,
        "packet_count": summary.packet_count,
        "unit_count": summary.unit_count,
    }
    if summary.causes is not None:
        view["causes"] = [cause.to_json_object() for cause in summary.causes]
    return view


def build_move_cause_diff(
    baseline_entry: dict[str, object],
    candidate_entry: dict[str, object],
    baseline_source: str,
    candidate_source: str,
) -> dict[str, object] | None:
    """Builds a semantic diff of compiler-classified move causes."""
    baseline = parse_compile_report_move_causes(baseline_entry, baseline_source)
    candidate = parse_compile_report_move_causes(candidate_entry, candidate_source)
    if baseline is None and candidate is None:
        return None
    if baseline is None or candidate is None:
        return {
            "availability": {
                "baseline": "available" if baseline is not None else "unavailable",
                "candidate": "available" if candidate is not None else "unavailable",
            }
        }

    view: dict[str, object] = {
        "summary": {
            "kind_count": _count_change(baseline.kind_count, candidate.kind_count),
            "packet_count": _count_change(
                baseline.packet_count, candidate.packet_count
            ),
            "unit_count": _count_change(baseline.unit_count, candidate.unit_count),
        }
    }
    if baseline.causes is None or candidate.causes is None:
        view["cause_availability"] = {
            "baseline": "available" if baseline.causes is not None else "unavailable",
            "candidate": (
                "available" if candidate.causes is not None else "unavailable"
            ),
        }
        return view

    baseline_by_cause = {cause.cause: cause for cause in baseline.causes}
    candidate_by_cause = {cause.cause: cause for cause in candidate.causes}
    rows = []
    changed_count = 0
    unchanged_count = 0
    for cause_name in sorted(set(baseline_by_cause) | set(candidate_by_cause)):
        baseline_cause = baseline_by_cause.get(cause_name)
        candidate_cause = candidate_by_cause.get(cause_name)
        if baseline_cause is None:
            changed_count += 1
            rows.append(
                {
                    "cause": cause_name,
                    "status": "added",
                    "candidate": candidate_cause.to_json_object(),
                }
            )
            continue
        if candidate_cause is None:
            changed_count += 1
            rows.append(
                {
                    "cause": cause_name,
                    "status": "removed",
                    "baseline": baseline_cause.to_json_object(),
                }
            )
            continue
        status = (
            "unchanged"
            if (
                baseline_cause.packet_count == candidate_cause.packet_count
                and baseline_cause.unit_count == candidate_cause.unit_count
            )
            else "changed"
        )
        if status == "changed":
            changed_count += 1
        else:
            unchanged_count += 1
        rows.append(
            {
                "cause": cause_name,
                "status": status,
                "packet_count": _count_change(
                    baseline_cause.packet_count,
                    candidate_cause.packet_count,
                ),
                "unit_count": _count_change(
                    baseline_cause.unit_count,
                    candidate_cause.unit_count,
                ),
            }
        )
    view["changed_cause_count"] = changed_count
    view["unchanged_cause_count"] = unchanged_count
    view["causes"] = rows
    return view


def move_cause_diff_has_changes(diff: dict[str, object] | None) -> bool:
    """Returns whether a move-cause diff contains any changed evidence."""
    if diff is None:
        return False
    for key in ("availability", "cause_availability"):
        availability = diff.get(key)
        if isinstance(availability, dict) and (
            availability["baseline"] != availability["candidate"]
        ):
            return True
    if "summary" not in diff:
        return False
    for key, change_value in _require_object(diff["summary"], "summary").items():
        change = _require_object(change_value, f"summary.{key}")
        if change["baseline"] != change["candidate"]:
            return True
    return bool(diff.get("changed_cause_count", 0))


def append_move_cause_show_text(
    lines: list[str],
    move_causes: dict[str, object],
) -> None:
    """Appends human-readable move-cause evidence for one entry."""
    lines.append("  Move causes (compiler analysis)")
    lines.append(
        f"    total: {move_causes['packet_count']} packets, "
        f"{move_causes['unit_count']} register units across "
        f"{move_causes['kind_count']} causes"
    )
    causes = move_causes.get("causes")
    if causes is None:
        lines.append("    cause breakdown: unavailable")
        return
    for cause_value in _require_list(causes, "causes"):
        cause = _require_object(cause_value, "cause")
        lines.append(
            f"    {_format_cause_name(cause['cause'])}: "
            f"{cause['packet_count']} packets, {cause['unit_count']} units"
        )


def append_move_cause_diff_text(
    lines: list[str],
    move_causes: dict[str, object],
) -> None:
    """Appends a human-readable move-cause diff for one entry."""
    lines.append("  Move causes (compiler analysis)")
    availability = move_causes.get("availability")
    if isinstance(availability, dict):
        lines.append(
            f"    evidence: {availability['baseline']} -> {availability['candidate']}"
        )
        return
    summary = _require_object(move_causes["summary"], "summary")
    for key, label in (
        ("kind_count", "cause kinds"),
        ("packet_count", "packets"),
        ("unit_count", "register units"),
    ):
        change = _require_object(summary[key], f"summary.{key}")
        lines.append(f"    {label}: {_format_count_change(change)}")
    cause_availability = move_causes.get("cause_availability")
    if isinstance(cause_availability, dict):
        lines.append(
            "    cause breakdown: "
            f"{cause_availability['baseline']} -> "
            f"{cause_availability['candidate']}"
        )
        return
    lines.append(
        f"    causes: {move_causes['changed_cause_count']} changed, "
        f"{move_causes['unchanged_cause_count']} unchanged"
    )
    for cause_value in _require_list(move_causes["causes"], "causes"):
        cause = _require_object(cause_value, "cause")
        label = _format_cause_name(cause["cause"])
        status = cause["status"]
        if status == "added":
            candidate = _require_object(cause["candidate"], "cause.candidate")
            lines.append(
                f"    {label}: added {candidate['packet_count']} packets, "
                f"{candidate['unit_count']} units"
            )
        elif status == "removed":
            baseline = _require_object(cause["baseline"], "cause.baseline")
            lines.append(
                f"    {label}: removed {baseline['packet_count']} packets, "
                f"{baseline['unit_count']} units"
            )
        else:
            packet_change = _require_object(cause["packet_count"], "cause.packet_count")
            unit_change = _require_object(cause["unit_count"], "cause.unit_count")
            lines.append(
                f"    {label}: {_format_count_change(packet_change)} packets, "
                f"{_format_count_change(unit_change)} units"
            )


def _count_change(baseline: int, candidate: int) -> dict[str, object]:
    change: dict[str, object] = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate - baseline,
    }
    if baseline != 0:
        change["change_percent"] = (candidate - baseline) * 100.0 / baseline
    return change


def _format_count_change(change: dict[str, object]) -> str:
    baseline = change["baseline"]
    candidate = change["candidate"]
    if baseline == candidate:
        return f"unchanged at {baseline}"
    return f"{baseline} -> {candidate}, delta {int(change['delta']):+d}"


def _format_cause_name(value: object) -> str:
    return _require_string(value, "cause").replace("_", " ")


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{source}: expected object")
    return value


def _require_list(value: object, source: str) -> list[object]:
    if not isinstance(value, list):
        raise CompileReportError(f"{source}: expected array")
    return value


def _require_string(value: object, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{source}: expected non-empty string")
    return value


def _require_count(value: object, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CompileReportError(f"{source}: expected non-negative integer")
    return value
