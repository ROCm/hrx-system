# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.tilelang.coverage import (
    TILELANG_OP_COVERAGE,
    CoverageState,
    audit_coverage,
    coverage_by_name,
)


def test_coverage_manifest_has_unique_names() -> None:
    coverage = coverage_by_name()

    assert len(coverage) == len(TILELANG_OP_COVERAGE)


def test_coverage_manifest_has_representative_states() -> None:
    states = {row.state for row in TILELANG_OP_COVERAGE}

    assert CoverageState.SUPPORTED in states
    assert CoverageState.NORMALIZED in states
    assert CoverageState.DEFERRED in states
    assert CoverageState.REJECTED in states


def test_audit_coverage_reports_missing_names() -> None:
    audit = audit_coverage(
        [
            "tir.For",
            "tl.tileop.copy",
            "tl.tileop.custom",
            "tl.future_op",
        ]
    )

    assert audit.known == ("tir.For", "tl.tileop.copy")
    assert audit.missing == ("tl.future_op", "tl.tileop.custom")
