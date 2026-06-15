# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.diagnostics import (
    DiagnosticEngine,
    DiagnosticParam,
    DiagnosticRelatedLocation,
    LoomDiagnosticError,
    SourceRange,
)


def test_diagnostic_engine_raises_only_for_errors() -> None:
    diagnostics = DiagnosticEngine()
    diagnostics.warning("watch this")

    diagnostics.raise_if_errors()

    diagnostics.error("failed", source="arith.weird")
    error_message = None
    try:
        diagnostics.raise_if_errors()
    except LoomDiagnosticError as exc:
        error_message = str(exc)
    if error_message is None:
        raise AssertionError("expected diagnostic errors to raise")
    assert "arith.weird" in error_message


def test_diagnostic_engine_records_source_ranges_and_related_locations() -> None:
    diagnostics = DiagnosticEngine()
    diagnostics.error(
        "failed",
        source_location=SourceRange.line(Path("model.loom"), 42, column=15),
        related_locations=(
            DiagnosticRelatedLocation(
                label="defined here",
                source_location=SourceRange.line(Path("model.loom"), 7, column=3),
            ),
        ),
    )

    diagnostic = diagnostics.diagnostics[0]

    assert diagnostic.primary_location is not None
    assert diagnostic.primary_location.display() == "model.loom:42:15"
    assert "defined here" in str(diagnostic)


def test_diagnostic_renders_params_for_structured_matching() -> None:
    diagnostics = DiagnosticEngine()
    diagnostics.error(
        "failed",
        params=(
            DiagnosticParam("flag", True),
            DiagnosticParam("shape", (1, 2, 4)),
        ),
    )

    assert diagnostics.diagnostics[0].rendered_params() == {
        "flag": "true",
        "shape": "1, 2, 4",
    }
