# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar target-legalization events retained by Loom compile reports."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import (
    CompileReportDocument,
    CompileReportError,
    compile_report_entry_identity,
)
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionEvidence,
)

_PATH = "target_legalization"

_ACTION_COUNT_FIELDS = {
    "legal": "legal_op_count",
    "rewritten": "rewritten_op_count",
    "deferred": "deferred_op_count",
    "reject-invalid-ir": "invalid_ir_op_count",
    "reject-unsupported-final": "unsupported_op_count",
    "unhandled": "unhandled_op_count",
}

_LEGALIZER_STRATEGIES = frozenset(("none", "target", "reference"))


@dataclass(frozen=True, slots=True)
class CompileReportScalarizationEvent:
    """One source operation expanded through scalar legalization."""

    row_index: int
    function: str
    source_op: str
    legalizer: str
    legalizer_strategy: str
    created_op_count: int
    erased_op_count: int

    @property
    def net_op_growth(self) -> int:
        """Returns the operation-count delta across the source rewrite lineage."""
        return self.created_op_count - self.erased_op_count


@dataclass(frozen=True, slots=True)
class CompileReportScalarizationInventory:
    """Validated scalarization summary plus detailed source events."""

    legalization_op_count: int
    rewritten_op_count: int
    scalarized_op_count: int
    events: tuple[CompileReportScalarizationEvent, ...]


def parse_compile_report_scalarization(
    report: dict[str, object], mode: str, source: str
) -> CompileReportScalarizationInventory | None:
    """Validates and indexes scalarization facts at the report boundary."""
    value = report.get(_PATH)
    if value is None:
        return None
    path = f"{source}.{_PATH}"
    section = _object(value, path)
    action_counts = {
        action: _count(section.get(field), f"{path}.{field}")
        for action, field in _ACTION_COUNT_FIELDS.items()
    }
    legalization_op_count = sum(action_counts.values())
    rewritten_op_count = action_counts["rewritten"]
    scalarized_op_count = _count(
        section.get("scalarized_op_count"), f"{path}.scalarized_op_count"
    )
    if scalarized_op_count > rewritten_op_count:
        raise CompileReportError(
            f"{path}.scalarized_op_count: exceeds rewritten operation count"
        )
    target_rewritten_op_count = _count(
        section.get("target_rewritten_op_count"),
        f"{path}.target_rewritten_op_count",
    )
    reference_rewritten_op_count = _count(
        section.get("reference_rewritten_op_count"),
        f"{path}.reference_rewritten_op_count",
    )
    if target_rewritten_op_count + reference_rewritten_op_count != rewritten_op_count:
        raise CompileReportError(
            f"{path}: target and reference rewrite counts do not sum to "
            "rewritten_op_count"
        )

    row_count = _count(section.get("count"), f"{path}.count")
    rows_value = section.get("rows")
    if mode == "summary":
        if rows_value is not None:
            raise CompileReportError(f"{path}.rows: unexpected in summary report")
        return CompileReportScalarizationInventory(
            legalization_op_count=legalization_op_count,
            rewritten_op_count=rewritten_op_count,
            scalarized_op_count=scalarized_op_count,
            events=(),
        )
    if rows_value is None:
        raise CompileReportError(f"{path}.rows: detailed report is missing rows")
    rows = _array(rows_value, f"{path}.rows")
    if row_count != len(rows):
        raise CompileReportError(f"{path}.count: does not match legalization rows")
    if row_count != legalization_op_count:
        raise CompileReportError(f"{path}.count: does not match action counts")

    events = []
    actual_action_counts = {action: 0 for action in _ACTION_COUNT_FIELDS}
    actual_rewrite_strategy_counts = {"target": 0, "reference": 0}
    for position, value in enumerate(rows):
        row_path = f"{path}.rows[{position}]"
        row = _object(value, row_path)
        index = _count(row.get("index"), f"{row_path}.index")
        if index != position:
            raise CompileReportError(f"{row_path}.index: expected {position}")
        function = _optional_string(row.get("function"), f"{row_path}.function")
        source_op = _optional_string(row.get("source_op"), f"{row_path}.source_op")
        legalizer = _optional_string(row.get("legalizer"), f"{row_path}.legalizer")
        legalizer_strategy = _choice(
            row.get("legalizer_strategy"),
            f"{row_path}.legalizer_strategy",
            _LEGALIZER_STRATEGIES,
        )
        action = _choice(
            row.get("action"),
            f"{row_path}.action",
            frozenset(_ACTION_COUNT_FIELDS),
        )
        scalarized = _boolean(row.get("scalarized"), f"{row_path}.scalarized")
        created_op_count = _count(
            row.get("created_op_count"), f"{row_path}.created_op_count"
        )
        erased_op_count = _count(
            row.get("erased_op_count"), f"{row_path}.erased_op_count"
        )
        actual_action_counts[action] += 1
        if (
            action == "rewritten"
            and legalizer_strategy in actual_rewrite_strategy_counts
        ):
            actual_rewrite_strategy_counts[legalizer_strategy] += 1

        if not scalarized:
            continue
        if action != "rewritten":
            raise CompileReportError(
                f"{row_path}.scalarized: scalarization must be a rewrite"
            )
        if function is None or source_op is None or legalizer is None:
            raise CompileReportError(
                f"{row_path}.scalarized: missing source or legalizer identity"
            )
        events.append(
            CompileReportScalarizationEvent(
                row_index=position,
                function=function,
                source_op=source_op,
                legalizer=legalizer,
                legalizer_strategy=legalizer_strategy,
                created_op_count=created_op_count,
                erased_op_count=erased_op_count,
            )
        )

    for action, expected_count in action_counts.items():
        if actual_action_counts[action] != expected_count:
            field = _ACTION_COUNT_FIELDS[action]
            raise CompileReportError(f"{path}.{field}: does not match rows")
    for strategy, expected_count in (
        ("target", target_rewritten_op_count),
        ("reference", reference_rewritten_op_count),
    ):
        if actual_rewrite_strategy_counts[strategy] != expected_count:
            raise CompileReportError(
                f"{path}.{strategy}_rewritten_op_count: does not match rows"
            )
    if len(events) != scalarized_op_count:
        raise CompileReportError(
            f"{path}.scalarized_op_count: does not match scalarized rows"
        )
    return CompileReportScalarizationInventory(
        legalization_op_count=legalization_op_count,
        rewritten_op_count=rewritten_op_count,
        scalarized_op_count=scalarized_op_count,
        events=tuple(events),
    )


def build_scalarization_show(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds a prominent view of source operations expanded to scalars."""
    inventory = document.scalarization_inventory
    if inventory is None or inventory.scalarized_op_count == 0:
        return None
    view: dict[str, object] = {
        "legalization_op_count": inventory.legalization_op_count,
        "rewritten_op_count": inventory.rewritten_op_count,
        "scalarized_op_count": inventory.scalarized_op_count,
    }
    if inventory.events:
        view["rows"] = [
            _event_json(event)
            for event in sorted(
                inventory.events,
                key=lambda event: (-event.net_op_growth, event.row_index),
            )
        ]
    return view


def append_scalarization_show_text(lines: list[str], view: dict[str, object]) -> None:
    """Appends scalar legalization warnings to a report show view."""
    lines.extend(
        (
            "",
            "Scalar lane expansion",
            (
                f"  scalarized={view['scalarized_op_count']} "
                f"rewritten={view['rewritten_op_count']} "
                f"legalization_ops={view['legalization_op_count']}"
            ),
        )
    )
    lines.extend(
        (
            f"  {row['function']} {row['source_op']}: "
            f"{row['created_op_count']} created, {row['erased_op_count']} erased "
            f"({int(row['net_op_growth']):+d} net), rooted at "
            f"{row['legalizer']}/{row['legalizer_strategy']}"
        )
        for row in _rows(view.get("rows", []))
    )


def suggest_scalarization(
    document: CompileReportDocument,
) -> tuple[CompileReportSuggestion, ...]:
    """Emits one high-confidence finding for every scalarized source op."""
    inventory = document.scalarization_inventory
    if inventory is None or inventory.scalarized_op_count == 0:
        return ()
    if not inventory.events:
        entry_name = _summary_entry_name(document)
        operation_noun = (
            "operation" if inventory.scalarized_op_count == 1 else "operations"
        )
        return (
            CompileReportSuggestion(
                suggestion_id="vector.inspect_scalarization",
                entry_name=entry_name,
                action=(
                    f"{inventory.scalarized_op_count} source {operation_noun} "
                    "expanded through scalar target legalization. Regenerate "
                    "this compilation with --compile-report=details to identify "
                    "each source operation, then replace its scalar fallback "
                    "with a target-vector lowering."
                ),
                evidence=(
                    CompileReportSuggestionEvidence(
                        f"{_PATH}.scalarized_op_count",
                        inventory.scalarized_op_count,
                    ),
                    CompileReportSuggestionEvidence(
                        f"{_PATH}.rewritten_op_count",
                        inventory.rewritten_op_count,
                    ),
                ),
            ),
        )

    suggestions = []
    for event in sorted(
        inventory.events,
        key=lambda event: (-event.net_op_growth, event.row_index),
    ):
        path = f"{_PATH}.rows[{event.row_index}]"
        suggestions.append(
            CompileReportSuggestion(
                suggestion_id="vector.eliminate_scalarization",
                entry_name=event.function,
                action=(
                    f"The rewrite rooted at {event.source_op} in "
                    f"{event.function} created {event.created_op_count} "
                    f"operations and erased {event.erased_op_count} "
                    f"({event.net_op_growth:+d} net) while expanding vector "
                    "semantics into scalar lanes. Keep the rewrite lineage "
                    "vector-native, recompile until scalarized=false, then "
                    "compare final code size, register pressure, spills, "
                    "instruction count, and measured runtime with the workload "
                    "held fixed."
                ),
                evidence=tuple(
                    CompileReportSuggestionEvidence(f"{path}.{field}", value)
                    for field, value in (
                        ("scalarized", True),
                        ("source_op", event.source_op),
                        ("legalizer", event.legalizer),
                        ("legalizer_strategy", event.legalizer_strategy),
                        ("created_op_count", event.created_op_count),
                        ("erased_op_count", event.erased_op_count),
                    )
                ),
            )
        )
    return tuple(suggestions)


def _summary_entry_name(document: CompileReportDocument) -> str:
    for field in ("target_export", "function", "module"):
        value = document.report.get(field)
        if isinstance(value, str) and value:
            return value
    if len(document.entries) == 1:
        return compile_report_entry_identity(document.entries[0]).display_name()
    return "<report>"


def _event_json(event: CompileReportScalarizationEvent) -> dict[str, object]:
    return {
        "index": event.row_index,
        "function": event.function,
        "source_op": event.source_op,
        "legalizer": event.legalizer,
        "legalizer_strategy": event.legalizer_strategy,
        "created_op_count": event.created_op_count,
        "erased_op_count": event.erased_op_count,
        "net_op_growth": event.net_op_growth,
    }


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise CompileReportError(f"{path}: expected array")
    return value


def _count(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompileReportError(f"{path}: expected nonnegative integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise CompileReportError(f"{path}: expected boolean")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{path}: expected nonempty string or null")
    return value


def _choice(value: object, path: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise CompileReportError(f"{path}: unsupported value")
    return value


def _rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError("invalid internal scalarization row array")
    return value
