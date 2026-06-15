# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.support.c import (
    CIdentifierCase,
    c_identifier,
    c_identifier_parts,
    c_pascal_identifier,
    c_string_arg,
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
