# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.support.c import (
    CIdentifierCase,
    c_i64_literal,
    c_identifier,
    c_identifier_parts,
    c_pascal_identifier,
    c_string_arg,
    c_string_classifier_lines,
    c_string_literal,
    c_string_view,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("amdgpu.buffer.atomic", ("amdgpu", "buffer", "atomic")),
        ("gfx12_0/gfx12-1", ("gfx12", "0", "gfx12", "1")),
        ("...", ()),
    ],
)
def test_c_identifier_parts(value: str, expected: tuple[str, ...]) -> None:
    assert c_identifier_parts(value) == expected


def test_c_identifier_preserves_requested_case() -> None:
    assert c_identifier("low.asm.optional") == "low_asm_optional"
    assert c_identifier("9lives") == "_9lives"
    assert c_identifier("...", empty="empty") == "empty"
    assert c_identifier("Mixed.Case", case=CIdentifierCase.LOWER) == "mixed_case"
    assert c_identifier("Mixed.Case", case=CIdentifierCase.UPPER) == "MIXED_CASE"


@pytest.mark.parametrize("keyword", ["requires", "class", "switch"])
def test_c_identifier_escapes_c_and_cpp_keywords(keyword: str) -> None:
    assert c_identifier(keyword) == f"{keyword}_"


def test_c_identifier_escapes_after_case_conversion() -> None:
    assert c_identifier("CLASS", case=CIdentifierCase.LOWER) == "class_"
    assert c_identifier("class", case=CIdentifierCase.UPPER) == "CLASS"


def test_c_identifier_rejects_empty_replacement() -> None:
    with pytest.raises(ValueError, match="empty replacement"):
        c_identifier("...", empty="")


def test_c_pascal_identifier() -> None:
    assert c_pascal_identifier("amdgpu.buffer_atomic") == "AmdgpuBufferAtomic"


def test_c_string_literal_escapes_c_control_characters() -> None:
    assert c_string_literal('a\\b"c\n\r\t') == 'a\\\\b\\"c\\n\\r\\t'
    assert c_string_arg("hello") == '"hello"'
    assert c_string_view("hello") == 'IREE_SVL("hello")'
    assert c_string_view("hello", macro="LOOM_SV") == 'LOOM_SV("hello")'


def test_c_string_classifier_lines_partitions_by_length() -> None:
    assert c_string_classifier_lines(
        (("b", "RESULT_B"), ("aa", "RESULT_AA"), ("a", "RESULT_A")),
        input_name="name",
        unmatched_result="RESULT_NONE",
        indent="  ",
    ) == [
        "  switch (name.size) {",
        "    case 1:",
        '      if (iree_string_view_equal(name, IREE_SV("b"))) {',
        "        return RESULT_B;",
        "      }",
        '      if (iree_string_view_equal(name, IREE_SV("a"))) {',
        "        return RESULT_A;",
        "      }",
        "      break;",
        "    case 2:",
        '      if (iree_string_view_equal(name, IREE_SV("aa"))) {',
        "        return RESULT_AA;",
        "      }",
        "      break;",
        "  }",
        "  return RESULT_NONE;",
    ]


def test_c_string_classifier_lines_rejects_duplicate_spelling() -> None:
    with pytest.raises(ValueError, match=r"duplicate.*'same'"):
        c_string_classifier_lines(
            (("same", "RESULT_A"), ("same", "RESULT_B")),
            input_name="name",
            unmatched_result="RESULT_NONE",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "INT64_C(0)"),
        (1 << 31, "INT64_C(2147483648)"),
        (-1, "(-INT64_C(1))"),
        (-(1 << 31), "(-INT64_C(2147483648))"),
        (-(1 << 63), "INT64_MIN"),
    ],
)
def test_c_i64_literal(value: int, expected: str) -> None:
    assert c_i64_literal(value) == expected


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 63])
def test_c_i64_literal_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        c_i64_literal(value)
