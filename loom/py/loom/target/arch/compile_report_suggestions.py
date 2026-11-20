# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit target-architecture compile report suggestion providers."""

from __future__ import annotations

from loom.reporting.compile_report import CompileReportDocument
from loom.reporting.compile_report_boundary_projections import (
    suggest_boundary_projections,
)
from loom.reporting.compile_report_expansions import suggest_expansions
from loom.reporting.compile_report_loop_pipelines import suggest_loop_pipelines
from loom.reporting.compile_report_math_relaxations import (
    suggest_native_math_relaxations,
)
from loom.reporting.compile_report_scalarization import suggest_scalarization
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestionOptions,
    CompileReportSuggestionProvider,
    CompileReportSuggestionResult,
    suggest_short_external_runs,
    suggest_single_slot_ingress,
)
from loom.target.arch.amd.xdna.aie2p.compile_report_suggestions import (
    AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER,
)
from loom.target.arch.amdgpu.compile_report_suggestions import (
    AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER,
)
from loom.target.arch.x86.compile_report_suggestions import (
    X86_COMPILE_REPORT_SUGGESTION_PROVIDER,
)

_PROVIDERS: tuple[CompileReportSuggestionProvider, ...] = (
    AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER,
    AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER,
    X86_COMPILE_REPORT_SUGGESTION_PROVIDER,
)


def suggest_compile_report(
    document: CompileReportDocument,
    options: CompileReportSuggestionOptions | None = None,
) -> CompileReportSuggestionResult:
    """Combines shared source evidence with the exact target-family provider."""
    if options is None:
        options = CompileReportSuggestionOptions()
    source_suggestions = (
        suggest_scalarization(document)
        + suggest_loop_pipelines(document)
        + suggest_boundary_projections(document)
        + suggest_native_math_relaxations(document)
    )
    if options.include_experimental and document.status_code == 0:
        source_suggestions += suggest_single_slot_ingress(document)
        source_suggestions += suggest_short_external_runs(document)
    target_result = _suggest_target(document, options)
    expansions = suggest_expansions(document)
    if not source_suggestions and not expansions:
        return target_result
    return CompileReportSuggestionResult(
        provider_name=(
            f"source+{target_result.provider_name}"
            if target_result.unavailable_reason is None
            else "source"
        ),
        unavailable_reason=None,
        suggestions=source_suggestions + target_result.suggestions + expansions,
        target_unavailable_reason=target_result.unavailable_reason,
    )


def _suggest_target(
    document: CompileReportDocument,
    options: CompileReportSuggestionOptions,
) -> CompileReportSuggestionResult:
    if document.status_code != 0:
        return CompileReportSuggestionResult(
            provider_name=None,
            unavailable_reason="compile_status_not_ok",
        )
    target_family = document.report.get("target_family")
    if target_family is None:
        return CompileReportSuggestionResult(
            provider_name=None,
            unavailable_reason="missing_target_family",
        )
    for provider in _PROVIDERS:
        if provider.target_family == target_family:
            return provider.suggest(document, options)
    return CompileReportSuggestionResult(
        provider_name=None,
        unavailable_reason="unsupported_target_family",
    )
