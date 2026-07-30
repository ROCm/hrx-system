# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit target-architecture compile report suggestion providers."""

from __future__ import annotations

from loom.reporting.compile_report import CompileReportDocument
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestionOptions,
    CompileReportSuggestionProvider,
    CompileReportSuggestionResult,
)
from loom.target.arch.amdgpu.compile_report_suggestions import (
    AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER,
)

_PROVIDERS: tuple[CompileReportSuggestionProvider, ...] = (
    AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER,
)


def suggest_compile_report(
    document: CompileReportDocument,
    options: CompileReportSuggestionOptions | None = None,
) -> CompileReportSuggestionResult:
    """Dispatches a validated report to its exact target-family provider."""
    if options is None:
        options = CompileReportSuggestionOptions()
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
