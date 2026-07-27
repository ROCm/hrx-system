# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for C array emission helpers."""

from __future__ import annotations

from loom.gen.support import c_arrays


def test_append_value_array_skips_empty() -> None:
    lines: list[str] = []

    c_arrays.append_value_array(lines, "uint16_t", "kValues", [])

    assert lines == []


def test_append_value_array() -> None:
    lines: list[str] = []

    c_arrays.append_value_array(lines, "uint16_t", "kValues", ["1", "2"])

    assert "\n".join(lines) == "\n".join(
        [
            "static const uint16_t kValues[] = {",
            "    1,",
            "    2,",
            "};",
            "",
        ]
    )


def test_append_value_array_without_trailing_blank() -> None:
    lines: list[str] = []

    c_arrays.append_value_array(
        lines,
        "uint16_t",
        "kValues",
        ["1"],
        trailing_blank=False,
    )

    assert "\n".join(lines) == "\n".join(
        [
            "static const uint16_t kValues[] = {",
            "    1,",
            "};",
        ]
    )


def test_append_struct_array_skips_empty() -> None:
    lines: list[str] = []

    c_arrays.append_struct_array(lines, "row_t", "kRows", [])

    assert lines == []


def test_append_struct_array() -> None:
    lines: list[str] = []

    c_arrays.append_struct_array(
        lines,
        "row_t",
        "kRows",
        [
            [".x = 1,", ".y = 2,"],
            [".x = 3,", ".y = 4,"],
        ],
    )

    assert "\n".join(lines) == "\n".join(
        [
            "static const row_t kRows[] = {",
            "    {",
            "        .x = 1,",
            "        .y = 2,",
            "    },",
            "    {",
            "        .x = 3,",
            "        .y = 4,",
            "    },",
            "};",
            "",
        ]
    )
