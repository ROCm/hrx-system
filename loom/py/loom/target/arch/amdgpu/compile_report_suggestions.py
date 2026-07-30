# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU compile report suggestions grounded in target-info rows."""

from __future__ import annotations

from loom.reporting.compile_report import (
    CompileReportDocument,
    compile_report_entry_identity,
)
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionEvidence,
    CompileReportSuggestionResult,
)
from loom.target.arch.amdgpu.target_identity import (
    AmdgpuArtifactTargetKeyError,
    parse_amdgpu_artifact_target_key,
)


class AmdgpuCompileReportSuggestionProvider:
    """Interprets AMDGPU evidence after resolving its structured target."""

    target_family = "AMDGPU"
    provider_name = "amdgpu"

    def suggest(self, document: CompileReportDocument) -> CompileReportSuggestionResult:
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


def _object_at(root: dict[str, object], *components: str) -> dict[str, object] | None:
    value: object = root
    for component in components:
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return value if isinstance(value, dict) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
