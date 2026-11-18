# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-native math forms exposed by explicit fast-math permissions."""

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

_PATH = "math_legalization"
_FASTMATH_FLAGS = (
    (1 << 0, "reassoc", "floating-point reassociation"),
    (1 << 1, "nnan", "assuming values are not NaNs"),
    (1 << 2, "ninf", "assuming values are not infinities"),
    (1 << 3, "nsz", "ignoring the sign of zero"),
    (1 << 4, "arcp", "reciprocal approximations"),
    (1 << 5, "contract", "floating-point contraction"),
    (1 << 6, "afn", "approximate math functions"),
)
_ALL_FASTMATH_FLAGS = sum(bit for bit, _, _ in _FASTMATH_FLAGS)
_MAX_REPORTED_MOTIFS = 8
_MAX_EVIDENCE_MOTIFS = 4


@dataclass(slots=True)
class CompileReportMathRelaxationMotif:
    """Repeated legalization rows sharing one native-form alternative."""

    first_row_index: int
    row_count: int
    source_op: str
    math_op: str
    lane_domain: str
    element_type: str
    recipe: str
    target_config: str | None
    native_required_fastmath_flags: int
    created_op_count: int


@dataclass(frozen=True, slots=True)
class CompileReportMathRelaxationGroup:
    """Native math relaxations sharing one source permission experiment."""

    function: str
    missing_fastmath_flags: int
    row_count: int
    motifs: tuple[CompileReportMathRelaxationMotif, ...]


@dataclass(frozen=True, slots=True)
class CompileReportMathRelaxationInventory:
    """Compact native-relaxation index retained at the report boundary."""

    report_row_count: int
    native_alternative_op_count: int
    groups: tuple[CompileReportMathRelaxationGroup, ...]


def parse_compile_report_math_relaxations(
    report: dict[str, object], mode: str, source: str
) -> CompileReportMathRelaxationInventory:
    """Validates and indexes explicit target-native math alternatives."""
    value = report.get(_PATH)
    if value is None:
        return CompileReportMathRelaxationInventory(0, 0, ())
    path = f"{source}.{_PATH}"
    section = _object(value, path)
    report_row_count = _count(section.get("count"), f"{path}.count")
    native_alternative_op_count = _count(
        section.get("native_alternative_op_count"),
        f"{path}.native_alternative_op_count",
    )
    rows_value = section.get("rows")
    if mode == "summary":
        if rows_value is not None:
            raise CompileReportError(f"{path}.rows: unexpected in summary report")
        return CompileReportMathRelaxationInventory(
            report_row_count, native_alternative_op_count, ()
        )
    rows = _array(rows_value, f"{path}.rows")
    if report_row_count != len(rows):
        raise CompileReportError(f"{path}.count: does not match legalization rows")

    groups: dict[
        tuple[str, int],
        dict[tuple[object, ...], CompileReportMathRelaxationMotif],
    ] = {}
    for position, value in enumerate(rows):
        row_path = f"{path}.rows[{position}]"
        row = _object(value, row_path)
        if _count(row.get("index"), f"{row_path}.index") != position:
            raise CompileReportError(f"{row_path}.index: expected {position}")
        required_value = row.get("native_required_fastmath_flags")
        if required_value is None:
            continue
        required_flags = _fastmath_flags(
            required_value, f"{row_path}.native_required_fastmath_flags"
        )
        if required_flags == 0:
            raise CompileReportError(
                f"{row_path}.native_required_fastmath_flags: expected nonzero flags"
            )
        source_flags = _fastmath_flags(
            row.get("source_fastmath_flags"), f"{row_path}.source_fastmath_flags"
        )
        missing_flags = required_flags & ~source_flags
        if missing_flags == 0:
            raise CompileReportError(
                f"{row_path}.native_required_fastmath_flags: already satisfied "
                "by source"
            )
        if _string(row.get("action"), f"{row_path}.action") != "rewritten":
            raise CompileReportError(
                f"{row_path}.native_required_fastmath_flags: expected a rewrite"
            )

        function = _string(row.get("function"), f"{row_path}.function")
        source_op = _string(row.get("source_op"), f"{row_path}.source_op")
        math_op = _string(row.get("math_op"), f"{row_path}.math_op")
        lane_domain = _string(row.get("lane_domain"), f"{row_path}.lane_domain")
        element_type = _string(row.get("element_type"), f"{row_path}.element_type")
        recipe = _string(row.get("recipe"), f"{row_path}.recipe")
        target_config = _optional_string(
            row.get("target_config"), f"{row_path}.target_config"
        )
        created_op_count = _count(
            row.get("created_op_count"), f"{row_path}.created_op_count"
        )
        key = (
            source_op,
            math_op,
            lane_domain,
            element_type,
            recipe,
            target_config,
            required_flags,
            created_op_count,
        )
        motifs = groups.setdefault((function, missing_flags), {})
        accumulator = motifs.get(key)
        if accumulator is None:
            motifs[key] = CompileReportMathRelaxationMotif(
                first_row_index=position,
                row_count=1,
                source_op=source_op,
                math_op=math_op,
                lane_domain=lane_domain,
                element_type=element_type,
                recipe=recipe,
                target_config=target_config,
                native_required_fastmath_flags=required_flags,
                created_op_count=created_op_count,
            )
        else:
            accumulator.row_count += 1

    result_groups = []
    for (function, missing_flags), accumulators in groups.items():
        motifs = tuple(
            sorted(accumulators.values(), key=lambda item: item.first_row_index)
        )
        result_groups.append(
            CompileReportMathRelaxationGroup(
                function=function,
                missing_fastmath_flags=missing_flags,
                row_count=sum(motif.row_count for motif in motifs),
                motifs=motifs,
            )
        )
    result_groups.sort(key=lambda group: group.motifs[0].first_row_index)
    retained_op_count = sum(group.row_count for group in result_groups)
    if retained_op_count != native_alternative_op_count:
        raise CompileReportError(
            f"{path}.native_alternative_op_count: does not match rows"
        )
    return CompileReportMathRelaxationInventory(
        report_row_count=report_row_count,
        native_alternative_op_count=native_alternative_op_count,
        groups=tuple(result_groups),
    )


def suggest_native_math_relaxations(
    document: CompileReportDocument,
) -> tuple[CompileReportSuggestion, ...]:
    """Suggests source flags only when target policy exposed a native form."""
    inventory = document.math_relaxation_inventory
    if inventory.native_alternative_op_count and not inventory.groups:
        operation = (
            "operation" if inventory.native_alternative_op_count == 1 else "operations"
        )
        return (
            CompileReportSuggestion(
                suggestion_id="math.inspect_native_fastmath",
                entry_name=_summary_entry_name(document),
                action=(
                    f"Target policy reports native forms for "
                    f"{inventory.native_alternative_op_count} source math {operation} "
                    "under additional fast-math permissions. Regenerate this "
                    "compilation with --compile-report=details to identify the "
                    "operations and exact permissions, then accept a relaxation "
                    "only if its floating-point contract and measured numerical "
                    "error are suitable for the workload."
                ),
                evidence=(
                    CompileReportSuggestionEvidence(
                        f"{_PATH}.native_alternative_op_count",
                        inventory.native_alternative_op_count,
                    ),
                ),
            ),
        )
    suggestions = []
    for group in inventory.groups:
        flag_names = _format_flag_names(group.missing_fastmath_flags)
        motif_summary = _describe_motifs(group.motifs)
        suggestions.append(
            CompileReportSuggestion(
                suggestion_id="math.use_native_fastmath",
                entry_name=group.function,
                variant=flag_names,
                action=(
                    f"Target policy reports native forms for {motif_summary} when "
                    f"{flag_names} is permitted. "
                    f"{_describe_flag_contract(group.missing_fastmath_flags)} "
                    f"Add {flag_names} to the cited "
                    "operations only if the relaxed floating-point contract is "
                    "acceptable, then recompile and validate numerical error "
                    "against the workload's tolerance. Compare final instructions "
                    "and measured runtime; native availability alone does not "
                    "establish a performance gain."
                ),
                evidence=_relaxation_evidence(group, inventory.report_row_count),
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


def _describe_motif(motif: CompileReportMathRelaxationMotif) -> str:
    operation = "operation" if motif.row_count == 1 else "operations"
    semantic_op = (
        ""
        if motif.source_op.rsplit(".", 1)[-1] == motif.math_op
        else f" `{motif.math_op}`"
    )
    return (
        f"{motif.row_count} {motif.lane_domain} {motif.element_type} "
        f"`{motif.source_op}`{semantic_op} {operation} expanded with "
        f"`{motif.recipe}`"
    )


def _describe_motifs(
    motifs: tuple[CompileReportMathRelaxationMotif, ...],
) -> str:
    visible = motifs[:_MAX_REPORTED_MOTIFS]
    summary = "; ".join(_describe_motif(motif) for motif in visible)
    remaining = motifs[_MAX_REPORTED_MOTIFS:]
    if remaining:
        remaining_rows = sum(motif.row_count for motif in remaining)
        summary += (
            f"; and {remaining_rows} operations across {len(remaining)} other motifs"
        )
    return summary


def _format_flag_names(flags: int) -> str:
    if flags == _ALL_FASTMATH_FLAGS:
        return "`fast`"
    names = [f"`{name}`" for bit, name, _ in _FASTMATH_FLAGS if flags & bit]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _describe_flag_contract(flags: int) -> str:
    effects = [description for bit, _, description in _FASTMATH_FLAGS if flags & bit]
    if len(effects) == 1:
        return f"This permits {effects[0]}."
    return f"Together these permit {', '.join(effects[:-1])}, and {effects[-1]}."


def _relaxation_evidence(
    group: CompileReportMathRelaxationGroup, report_row_count: int
) -> tuple[CompileReportSuggestionEvidence, ...]:
    evidence = []
    if group.row_count == report_row_count:
        evidence.append(
            CompileReportSuggestionEvidence(f"{_PATH}.count", report_row_count)
        )
    for motif in group.motifs[:_MAX_EVIDENCE_MOTIFS]:
        path = f"{_PATH}.rows[{motif.first_row_index}]"
        for field, value in (
            ("source_op", motif.source_op),
            ("math_op", motif.math_op),
            ("lane_domain", motif.lane_domain),
            ("element_type", motif.element_type),
            ("recipe", motif.recipe),
            ("native_required_fastmath_flags", motif.native_required_fastmath_flags),
            ("created_op_count", motif.created_op_count),
        ):
            evidence.append(CompileReportSuggestionEvidence(f"{path}.{field}", value))
        if motif.target_config is not None:
            evidence.append(
                CompileReportSuggestionEvidence(
                    f"{path}.target_config", motif.target_config
                )
            )
    return tuple(evidence)


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


def _fastmath_flags(value: object, path: str) -> int:
    flags = _count(value, path)
    if flags & ~_ALL_FASTMATH_FLAGS:
        raise CompileReportError(f"{path}: contains unknown fast-math flags")
    return flags


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{path}: expected nonempty string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)
