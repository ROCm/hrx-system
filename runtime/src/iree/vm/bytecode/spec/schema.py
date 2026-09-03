# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared version, scalar, and natural-layout vocabulary."""

import re
from collections.abc import Iterable
from typing import NamedTuple

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_C_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_ ]*")


class ScalarEncoding(NamedTuple):
    name: str
    c_type: str
    byte_length: int
    alignment: int


U8 = ScalarEncoding("u8", "uint8_t", 1, 1)
I16 = ScalarEncoding("i16", "int16_t", 2, 2)
U16 = ScalarEncoding("u16", "uint16_t", 2, 2)
I32 = ScalarEncoding("i32", "int32_t", 4, 4)
U32 = ScalarEncoding("u32", "uint32_t", 4, 4)
U64 = ScalarEncoding("u64", "uint64_t", 8, 8)


class Field(NamedTuple):
    name: str
    encoding: ScalarEncoding
    summary: str
    element_count: int = 1

    @property
    def byte_length(self) -> int:
        return self.encoding.byte_length * self.element_count


def place_fields(
    fields: Iterable[Field], *, initial_offset: int = 0
) -> tuple[int, ...]:
    """Places explicitly padded fields without introducing hidden bytes."""
    if initial_offset < 0:
        raise ValueError("initial field offset must be nonnegative")
    offsets = []
    field_names = set()
    offset = initial_offset
    for field in fields:
        encoding = field.encoding
        if (
            not _NAME_PATTERN.fullmatch(encoding.name)
            or not _C_TYPE_PATTERN.fullmatch(encoding.c_type)
            or encoding.byte_length <= 0
            or encoding.byte_length & (encoding.byte_length - 1)
            or encoding.alignment <= 0
            or encoding.alignment & (encoding.alignment - 1)
            or encoding.alignment > encoding.byte_length
        ):
            raise ValueError(f"{field.name}: invalid scalar encoding")
        if not _NAME_PATTERN.fullmatch(field.name):
            raise ValueError(f"invalid field name {field.name!r}")
        if field.name in field_names:
            raise ValueError(f"duplicate field name {field.name!r}")
        field_names.add(field.name)
        if not field.summary.strip() or field.element_count <= 0:
            raise ValueError(f"{field.name}: incomplete field declaration")
        if offset % field.encoding.alignment:
            raise ValueError(
                f"{field.name}: offset {offset} violates "
                f"{field.encoding.alignment}-byte alignment; add explicit padding"
            )
        offsets.append(offset)
        offset += field.byte_length
    return tuple(offsets)
