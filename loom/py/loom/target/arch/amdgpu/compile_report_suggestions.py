# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU compile report suggestions grounded in target-info rows."""

from __future__ import annotations

from loom.reporting.compile_report import (
    CompileReportDocument,
    CompileReportError,
    compile_report_entry_identity,
)
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionConfidence,
    CompileReportSuggestionEvidence,
    CompileReportSuggestionOptions,
    CompileReportSuggestionResult,
)
from loom.target.arch.amdgpu.lds_bank_service import (
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED,
)
from loom.target.arch.amdgpu.target_identity import (
    AmdgpuArtifactTargetKeyError,
    parse_amdgpu_artifact_target_key,
)


class AmdgpuCompileReportSuggestionProvider:
    """Interprets AMDGPU evidence after resolving its structured target."""

    target_family = "AMDGPU"
    provider_name = "amdgpu"

    def suggest(
        self,
        document: CompileReportDocument,
        options: CompileReportSuggestionOptions | None = None,
    ) -> CompileReportSuggestionResult:
        if options is None:
            options = CompileReportSuggestionOptions()
        target_key = document.report.get("target_key")
        if target_key is None:
            return CompileReportSuggestionResult(
                provider_name=self.provider_name,
                unavailable_reason="missing_target_key",
            )
        try:
            target_identity = parse_amdgpu_artifact_target_key(target_key)
        except AmdgpuArtifactTargetKeyError:
            return CompileReportSuggestionResult(
                provider_name=self.provider_name,
                unavailable_reason="invalid_target_key",
            )
        if target_identity is None:
            return CompileReportSuggestionResult(
                provider_name=self.provider_name,
                unavailable_reason="unknown_target_key",
            )

        suggestions = []
        for entry in document.entries:
            entry_index = entry["index"]
            entry_name = compile_report_entry_identity(entry).display_name()
            path_prefix = f"entries.rows[{entry_index}]"
            spill_suggestion = _suggest_spill_traffic(entry, entry_name, path_prefix)
            if spill_suggestion is not None:
                suggestions.append(spill_suggestion)
            else:
                private_memory_suggestion = _suggest_private_memory(
                    entry, entry_name, path_prefix
                )
                if private_memory_suggestion is not None:
                    suggestions.append(private_memory_suggestion)
            residency_suggestion = _suggest_residency_cliff(
                entry, entry_name, path_prefix
            )
            if residency_suggestion is not None:
                suggestions.append(residency_suggestion)
            wave_suggestion = _suggest_nondefault_wave_size(
                entry,
                entry_name,
                path_prefix,
                target_identity.processor.wavefront.default_size,
            )
            if wave_suggestion is not None:
                suggestions.append(wave_suggestion)
        suggestions.extend(_suggest_lds_bank_service(document, options))
        return CompileReportSuggestionResult(
            provider_name=self.provider_name,
            unavailable_reason=None,
            suggestions=tuple(suggestions),
        )


AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER = AmdgpuCompileReportSuggestionProvider()


def _suggest_spill_traffic(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    metrics = (
        ("allocation_spill_count", "allocation_spill_count"),
        (
            "allocation_materialized_spill_storage_bytes",
            "allocation_materialized_spill_storage_bytes",
        ),
        (
            "allocation_materialized_reload_bytes",
            "allocation_materialized_reload_bytes",
        ),
    )
    evidence = tuple(
        CompileReportSuggestionEvidence(
            path=f"{path_prefix}.{path}",
            value=value,
        )
        for field, path in metrics
        if (value := _integer(entry.get(field))) is not None and value > 0
    )
    if not evidence:
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.spill_traffic",
        entry_name=entry_name,
        action=(
            "Shorten live ranges or reduce fragment state, then require spill "
            "and reload evidence to fall before benchmarking."
        ),
        evidence=evidence,
    )


def _suggest_private_memory(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    private_memory_bytes = _integer(entry.get("private_memory_bytes"))
    if private_memory_bytes is None or private_memory_bytes == 0:
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.private_memory",
        entry_name=entry_name,
        action=(
            "Reduce per-workitem private storage and inspect the emitted scratch "
            "traffic before benchmarking."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.private_memory_bytes",
                value=private_memory_bytes,
            ),
        ),
    )


def _suggest_residency_cliff(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    residency = _object_at(entry, "target_resources", "residency")
    limiting_resource = _object_at(
        entry,
        "target_resources",
        "residency",
        "unique_limiting_resource",
    )
    if residency is None or limiting_resource is None:
        return None
    current_tier = _integer(residency.get("current_tier"))
    next_better_tier = _integer(residency.get("next_better_tier"))
    reduction = _integer(limiting_resource.get("reduction_units_to_next_better_tier"))
    resource_name = limiting_resource.get("name")
    if (
        current_tier is None
        or next_better_tier is None
        or next_better_tier <= current_tier
        or reduction is None
        or reduction == 0
        or not isinstance(resource_name, str)
    ):
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.residency_cliff",
        entry_name=entry_name,
        action=(
            f"Reduce {resource_name} use by at least {reduction} units, then "
            f"recompile and benchmark the predicted tier {next_better_tier}."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.target_resources.residency.current_tier",
                value=current_tier,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.target_resources.residency.next_better_tier",
                value=next_better_tier,
            ),
            CompileReportSuggestionEvidence(
                path=(
                    f"{path_prefix}.target_resources.residency."
                    "unique_limiting_resource.name"
                ),
                value=resource_name,
            ),
            CompileReportSuggestionEvidence(
                path=(
                    f"{path_prefix}.target_resources.residency."
                    "unique_limiting_resource."
                    "reduction_units_to_next_better_tier"
                ),
                value=reduction,
            ),
        ),
    )


def _suggest_nondefault_wave_size(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
    default_wave_size: int,
) -> CompileReportSuggestion | None:
    target_resources = _object_at(entry, "target_resources")
    if target_resources is None:
        return None
    subgroup_size = _integer(target_resources.get("subgroup_size"))
    if (
        subgroup_size is None
        or subgroup_size == 0
        or subgroup_size == default_wave_size
    ):
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.nondefault_wave_size",
        entry_name=entry_name,
        action=(
            f"Benchmark an equivalent wave{default_wave_size} specialization and "
            f"retain wave{subgroup_size} only when measured faster."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.target_resources.subgroup_size",
                value=subgroup_size,
            ),
            CompileReportSuggestionEvidence(
                path="target_info.wavefront.default_size",
                value=default_wave_size,
            ),
        ),
    )


def _suggest_lds_bank_service(
    document: CompileReportDocument,
    options: CompileReportSuggestionOptions,
) -> tuple[CompileReportSuggestion, ...]:
    source_low = document.report.get("source_low")
    if source_low is None:
        return ()
    source_low_object = _report_object(source_low, "source_low")
    memory = source_low_object.get("memory")
    if memory is None:
        return ()
    memory_object = _report_object(memory, "source_low.memory")
    group_values = memory_object.get("bank_service_groups")
    if group_values is None:
        return ()
    if not isinstance(group_values, list):
        raise CompileReportError(
            "source_low.memory.bank_service_groups: expected array"
        )
    group_count = _report_integer(
        memory_object.get("bank_service_group_count"),
        "source_low.memory.bank_service_group_count",
    )
    if group_count != len(group_values):
        raise CompileReportError(
            "source_low.memory.bank_service_group_count: "
            f"expected {len(group_values)}, got {group_count}"
        )

    suggestions = []
    for group_position, group_value in enumerate(group_values):
        path_prefix = f"source_low.memory.bank_service_groups[{group_position}]"
        group = _report_object(group_value, path_prefix)
        report_index = _report_integer(group.get("index"), f"{path_prefix}.index")
        if report_index != group_position:
            raise CompileReportError(
                f"{path_prefix}.index: expected {group_position}, got {report_index}"
            )
        model = _report_object(group.get("model"), f"{path_prefix}.model")
        summary = _report_object(group.get("summary"), f"{path_prefix}.summary")
        structural = _report_object(
            summary.get("structural"),
            f"{path_prefix}.summary.structural",
        )

        exact_packet_count = _report_integer(
            summary.get("exact_packet_count"),
            f"{path_prefix}.summary.exact_packet_count",
        )
        unknown_packet_count = _report_integer(
            summary.get("unknown_packet_count"),
            f"{path_prefix}.summary.unknown_packet_count",
        )
        conflicted_packet_count = _report_integer(
            structural.get("conflicted_packet_count"),
            f"{path_prefix}.summary.structural.conflicted_packet_count",
        )
        extra_round_count = _report_integer(
            structural.get("extra_round_count"),
            f"{path_prefix}.summary.structural.extra_round_count",
        )
        maximum_request_multiplicity = _report_integer(
            structural.get("maximum_request_multiplicity"),
            (f"{path_prefix}.summary.structural.maximum_request_multiplicity"),
        )
        if (
            exact_packet_count == 0
            or unknown_packet_count != 0
            or conflicted_packet_count == 0
            or extra_round_count == 0
        ):
            continue

        model_evidence = _report_string(
            model.get("evidence"),
            f"{path_prefix}.model.evidence",
        )
        if model_evidence in (
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
        ):
            confidence = CompileReportSuggestionConfidence.HIGH
        elif (
            model_evidence
            == AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED
        ):
            if not options.include_experimental:
                continue
            confidence = CompileReportSuggestionConfidence.EXPERIMENTAL
        else:
            raise CompileReportError(
                f"{path_prefix}.model.evidence: unsupported evidence class "
                f"{model_evidence!r}"
            )

        function_name = _optional_report_string(
            group.get("function"),
            f"{path_prefix}.function",
        )
        source_op = _optional_report_string(
            group.get("source_op"),
            f"{path_prefix}.source_op",
        )
        source_root = _optional_report_string(
            group.get("source_root"),
            f"{path_prefix}.source_root",
        )
        source_root_argument_index_value = group.get("source_root_argument_index")
        if source_root_argument_index_value is not None:
            source_root_argument_index = _report_integer(
                source_root_argument_index_value,
                f"{path_prefix}.source_root_argument_index",
            )
            if source_root is None:
                source_root = f"arg{source_root_argument_index}"
        packet = _optional_report_string(
            group.get("packet"),
            f"{path_prefix}.packet",
        )
        entry_name = _entry_name_for_function(document, function_name)
        location = "/".join(
            value for value in (source_op, source_root, packet) if value is not None
        )
        suggestions.append(
            CompileReportSuggestion(
                suggestion_id="amdgpu.lds_bank_service",
                entry_name=entry_name,
                confidence=confidence,
                action=(
                    f"Search layout variants for {location or 'this LDS access'} "
                    "using pitch or padding, lane mapping, fragment layout, or "
                    "packet width to reduce exact structural extra rounds. "
                    "Recompile each candidate, reject spill or occupancy "
                    "regressions, and select only from hardware timing."
                ),
                evidence=(
                    CompileReportSuggestionEvidence(
                        path=f"{path_prefix}.summary.exact_packet_count",
                        value=exact_packet_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=(
                            f"{path_prefix}.summary.structural.conflicted_packet_count"
                        ),
                        value=conflicted_packet_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=(f"{path_prefix}.summary.structural.extra_round_count"),
                        value=extra_round_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=(
                            f"{path_prefix}.summary.structural."
                            "maximum_request_multiplicity"
                        ),
                        value=maximum_request_multiplicity,
                    ),
                    CompileReportSuggestionEvidence(
                        path=f"{path_prefix}.model.evidence",
                        value=model_evidence,
                    ),
                    CompileReportSuggestionEvidence(
                        path=f"{path_prefix}.model.revision",
                        value=_report_string(
                            model.get("revision"),
                            f"{path_prefix}.model.revision",
                        ),
                    ),
                ),
            )
        )
    return tuple(suggestions)


def _entry_name_for_function(
    document: CompileReportDocument,
    function_name: str | None,
) -> str:
    if function_name is not None:
        for entry in document.entries:
            if function_name in (
                entry.get("function"),
                entry.get("source_function"),
                entry.get("target_export"),
                entry.get("target_export_symbol"),
            ):
                return compile_report_entry_identity(entry).display_name()
        return function_name
    if len(document.entries) == 1:
        return compile_report_entry_identity(document.entries[0]).display_name()
    return "<report>"


def _report_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _report_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CompileReportError(f"{path}: expected integer")
    return value


def _report_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise CompileReportError(f"{path}: expected string")
    return value


def _optional_report_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _report_string(value, path)


def _object_at(root: dict[str, object], *components: str) -> dict[str, object] | None:
    value: object = root
    for component in components:
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return value if isinstance(value, dict) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
