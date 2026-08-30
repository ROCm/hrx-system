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


def test_static_array_emitter_reuses_exact_value_initializer() -> None:
    lines: list[str] = []
    emitter = c_arrays.StaticArrayEmitter(lines)

    first_symbol = emitter.append_value_array(
        "uint16_t",
        "kFirstValues",
        ["1", "2"],
    )
    second_symbol = emitter.append_value_array(
        "uint16_t",
        "kSecondValues",
        ["1", "2"],
    )

    assert first_symbol == "kFirstValues"
    assert second_symbol == "kFirstValues"
    assert "kFirstValues" in "\n".join(lines)
    assert "kSecondValues" not in "\n".join(lines)


def test_static_array_emitter_can_reuse_required_empty_array() -> None:
    lines: list[str] = []
    emitter = c_arrays.StaticArrayEmitter(lines)

    first_symbol = emitter.append_value_array(
        "uint16_t",
        "kFirstValues",
        [],
        emit_empty=True,
    )
    second_symbol = emitter.append_value_array(
        "uint16_t",
        "kSecondValues",
        [],
        emit_empty=True,
    )

    assert first_symbol == "kFirstValues"
    assert second_symbol == "kFirstValues"
    assert "\n".join(lines) == "\n".join(
        [
            "static const uint16_t kFirstValues[] = {",
            "};",
            "",
        ]
    )


def test_static_array_emitter_keeps_types_and_initializer_kinds_distinct() -> None:
    lines: list[str] = []
    emitter = c_arrays.StaticArrayEmitter(lines)

    value_symbol = emitter.append_value_array("uint16_t", "kValues", ["1"])
    other_type_symbol = emitter.append_value_array(
        "uint32_t",
        "kOtherTypeValues",
        ["1"],
    )
    struct_symbol = emitter.append_struct_array(
        "uint16_t",
        "kStructValues",
        [["1"]],
    )

    assert value_symbol == "kValues"
    assert other_type_symbol == "kOtherTypeValues"
    assert struct_symbol == "kStructValues"


def test_static_array_emitter_reuses_exact_struct_initializer() -> None:
    lines: list[str] = []
    emitter = c_arrays.StaticArrayEmitter(lines)
    rows = [[".x = 1,", ".y = 2,"]]

    first_symbol = emitter.append_struct_array("row_t", "kFirstRows", rows)
    second_symbol = emitter.append_struct_array(
        "row_t",
        "kSecondRows",
        rows,
    )

    assert first_symbol == "kFirstRows"
    assert second_symbol == "kFirstRows"
    assert "kFirstRows" in "\n".join(lines)
    assert "kSecondRows" not in "\n".join(lines)
