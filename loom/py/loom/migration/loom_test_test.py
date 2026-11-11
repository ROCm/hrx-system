# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for canonical ``.loom-test`` input-section rewriting."""

import pytest

from loom.migration.loom_test import rewrite_loom_test_inputs


def test_rewrites_only_case_inputs() -> None:
    source = (
        "// RUN: roundtrip\n"
        "\n"
        "first\n"
        "// ----\n"
        "expected first\n"
        "\n"
        "// ==== second\n"
        "// REQUIRES: target\n"
        "second\n"
        "// ----\n"
        "expected second\n"
    )

    result = rewrite_loom_test_inputs(
        source,
        lambda test_case, input_text: f"case {test_case.index}: {input_text}",
    )

    assert result.source == (
        "// RUN: roundtrip\n"
        "\n"
        "case 0: first\n"
        "// ----\n"
        "expected first\n"
        "\n"
        "// ==== second\n"
        "// REQUIRES: target\n"
        "case 1: second\n"
        "// ----\n"
        "expected second\n"
    )


def test_diagnostic_annotations_are_part_of_the_input() -> None:
    source = "// RUN: verify\n// ERROR@+1: PARSE/006\nbogus.op\n"

    result = rewrite_loom_test_inputs(
        source,
        lambda _test_case, input_text: input_text.replace("bogus.op", "other.op"),
    )

    assert result.source == ("// RUN: verify\n// ERROR@+1: PARSE/006\nother.op\n")


def test_empty_source_has_no_cases() -> None:
    with pytest.raises(ValueError, match="no cases"):
        rewrite_loom_test_inputs("\n", lambda _test_case, input_text: input_text)
