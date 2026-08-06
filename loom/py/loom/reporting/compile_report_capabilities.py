# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact semantic diffs for selected target capability rows."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportDocument, CompileReportError


@dataclass(frozen=True)
class _CapabilityIdentity:
    function: str | None
    target_family: str | None
    namespace: str
    key: str

    def to_json_object(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "namespace": self.namespace,
            "key": self.key,
        }
        if self.function is not None:
            identity["function"] = self.function
        if self.target_family is not None:
            identity["target_family"] = self.target_family
        return identity


@dataclass(frozen=True)
class _CapabilityValue:
    kind: str
    value: bool | int | str | None

    def to_json_object(self) -> dict[str, object]:
        return {"kind": self.kind, "value": self.value}


def build_target_capability_diff(
    baseline: CompileReportDocument,
    candidate: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds a semantic diff of capabilities selected for emitted entries."""
    baseline_capabilities = _parse_target_capabilities(baseline)
    candidate_capabilities = _parse_target_capabilities(candidate)
    if baseline_capabilities is None and candidate_capabilities is None:
        return None
    if baseline_capabilities is None or candidate_capabilities is None:
        return {
            "availability": {
                "baseline": (
                    "available" if baseline_capabilities is not None else "unavailable"
                ),
                "candidate": (
                    "available" if candidate_capabilities is not None else "unavailable"
                ),
            },
            "changed_count": 0,
            "unchanged_count": 0,
            "rows": [],
        }

    changed_rows = []
    unchanged_count = 0
    for identity in sorted(
        set(baseline_capabilities) | set(candidate_capabilities),
        key=_capability_identity_sort_key,
    ):
        baseline_capability = baseline_capabilities.get(identity)
        candidate_capability = candidate_capabilities.get(identity)
        if baseline_capability is None:
            changed_rows.append(
                {
                    "identity": identity.to_json_object(),
                    "status": "added",
                    "candidate": candidate_capability.to_json_object(),
                }
            )
            continue
        if candidate_capability is None:
            changed_rows.append(
                {
                    "identity": identity.to_json_object(),
                    "status": "removed",
                    "baseline": baseline_capability.to_json_object(),
                }
            )
            continue
        if baseline_capability == candidate_capability:
            unchanged_count += 1
            continue
        changed_rows.append(
            {
                "identity": identity.to_json_object(),
                "status": "changed",
                "baseline": baseline_capability.to_json_object(),
                "candidate": candidate_capability.to_json_object(),
            }
        )

    return {
        "changed_count": len(changed_rows),
        "unchanged_count": unchanged_count,
        "rows": changed_rows,
    }


def append_target_capability_diff_text(
    lines: list[str],
    capability_diff: dict[str, object],
) -> None:
    """Appends a compact human-readable capability diff."""
    lines.append("")
    lines.append("Target capabilities")
    availability = capability_diff.get("availability")
    if isinstance(availability, dict):
        lines.append(
            "  availability: "
            f"baseline={availability['baseline']}, "
            f"candidate={availability['candidate']}"
        )
        return
    lines.append(
        f"  rows: {capability_diff['changed_count']} changed, "
        f"{capability_diff['unchanged_count']} unchanged"
    )
    for row_value in _require_list(capability_diff.get("rows"), "rows"):
        row = _require_object(row_value, "row")
        identity = _require_object(row.get("identity"), "row.identity")
        function = identity.get("function")
        function_prefix = f"{function} " if function else ""
        label = f"{function_prefix}{identity['namespace']}.{identity['key']}"
        status = row.get("status")
        if status == "changed":
            baseline = _require_object(row.get("baseline"), "row.baseline")
            candidate = _require_object(row.get("candidate"), "row.candidate")
            lines.append(
                f"  {label}: {_format_value(baseline)} -> {_format_value(candidate)}"
            )
        elif status == "added":
            candidate = _require_object(row.get("candidate"), "row.candidate")
            lines.append(f"  {label}: added {_format_value(candidate)}")
        elif status == "removed":
            baseline = _require_object(row.get("baseline"), "row.baseline")
            lines.append(f"  {label}: removed {_format_value(baseline)}")
        else:
            raise CompileReportError(
                f"target capability row: invalid status {status!r}"
            )


def _parse_target_capabilities(
    document: CompileReportDocument,
) -> dict[_CapabilityIdentity, _CapabilityValue] | None:
    value = document.report.get("target_capability_rows")
    if value is None:
        return None
    container = _require_object(value, f"{document.source}.target_capability_rows")
    rows = _require_list(
        container.get("rows"), f"{document.source}.target_capability_rows.rows"
    )
    count = container.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(rows):
        raise CompileReportError(
            f"{document.source}.target_capability_rows.count: "
            f"expected {len(rows)}, got {count!r}"
        )

    capabilities: dict[_CapabilityIdentity, _CapabilityValue] = {}
    for index, row_value in enumerate(rows):
        source = f"{document.source}.target_capability_rows.rows[{index}]"
        row = _require_object(row_value, source)
        identity = _CapabilityIdentity(
            function=_optional_string(row.get("function"), f"{source}.function"),
            target_family=_optional_string(
                row.get("target_family"), f"{source}.target_family"
            ),
            namespace=_require_string(row.get("namespace"), f"{source}.namespace"),
            key=_require_string(row.get("key"), f"{source}.key"),
        )
        if identity in capabilities:
            raise CompileReportError(
                f"{source}: duplicate target capability "
                f"{identity.namespace}.{identity.key!s}"
            )
        capabilities[identity] = _parse_capability_value(row, source)
    return capabilities


def _capability_identity_sort_key(
    identity: _CapabilityIdentity,
) -> tuple[str, str, str, str]:
    return (
        identity.function or "",
        identity.target_family or "",
        identity.namespace,
        identity.key,
    )


def _parse_capability_value(
    row: dict[str, object],
    source: str,
) -> _CapabilityValue:
    kind = _require_string(row.get("value_kind"), f"{source}.value_kind")
    if kind == "none":
        value: bool | int | str | None = None
    elif kind == "bool":
        value = row.get("value_bool")
        if not isinstance(value, bool):
            raise CompileReportError(f"{source}.value_bool: expected bool")
    elif kind == "u64":
        value = row.get("value_u64")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CompileReportError(f"{source}.value_u64: expected unsigned integer")
    elif kind == "string":
        value = _require_string(row.get("value_string"), f"{source}.value_string")
    else:
        raise CompileReportError(f"{source}.value_kind: unsupported kind {kind!r}")
    return _CapabilityValue(kind=kind, value=value)


def _format_value(value: dict[str, object]) -> str:
    kind = value.get("kind")
    payload = value.get("value")
    if kind == "string":
        return repr(payload)
    if isinstance(payload, bool):
        return "true" if payload else "false"
    if payload is None:
        return "none"
    return str(payload)


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


def _optional_string(value: object, source: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, source)
