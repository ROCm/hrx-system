# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared version, scalar, and natural-layout vocabulary."""

import enum
import re
from collections.abc import Iterable
from typing import NamedTuple

from iree.vm.bytecode.spec.version import Version

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_name(value: str, *, qualified: bool = False) -> bool:
    return bool(_NAME_PATTERN.fullmatch(value)) and (qualified or "." not in value)


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


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


class NumericKind(enum.Enum):
    ENUM = "enum"
    FLAGS = "flags"
    SELECTOR = "selector"


class UnknownNumericPolicy(enum.Enum):
    REJECT = "reject"
    PRESERVE_NONZERO = "preserve_nonzero"


class NumericValue(NamedTuple):
    name: str
    value: int
    since: Version
    summary: str


class NumericTable(NamedTuple):
    name: str
    encoding: ScalarEncoding
    kind: NumericKind
    values: tuple[NumericValue, ...]
    since: Version
    summary: str
    unknown_policy: UnknownNumericPolicy = UnknownNumericPolicy.REJECT


class FieldRuleUse(NamedTuple):
    kind: "RuleKind"
    fields: tuple[str, ...] = ()
    values: tuple[int, ...] = ()
    data: object | None = None


def validate_numeric_table(table: NumericTable, version: Version) -> None:
    values = table.values
    valid = is_name(table.name, qualified=True) and bool(values)
    valid &= len({value.name for value in values}) == len(values)
    valid &= len({value.value for value in values}) == len(values)
    valid &= any(value.since == table.since for value in values)
    valid &= all(
        is_name(value.name, qualified=True)
        and 0 <= value.value < 1 << (table.encoding.byte_length * 8)
        and table.since.is_available_in(value.since)
        and value.since.is_available_in(version)
        and (
            table.kind != NumericKind.FLAGS
            or value.value == 0
            or is_power_of_two(value.value)
        )
        for value in values
    )
    require(valid, f"{table.name}: invalid numeric table")


class RuleKind(NamedTuple):
    name: str
    encoding: ScalarEncoding | None = None
    field_count: int = 0
    value_count: int = 0
    data_count: int = 0
    data_type: type | None = None
    summary: str = ""

    def accepts(self, fields, values, data=None) -> bool:
        return (
            len(fields) == self.field_count
            and (self.value_count < 0 or len(values) == self.value_count)
            and isinstance(data, self.data_type or type(None))
            and (not self.data_count or len(data or ()) == self.data_count)
        )


def place_fields(
    fields: Iterable[Field], *, initial_offset: int = 0
) -> tuple[int, ...]:
    """Places explicitly padded fields without introducing hidden bytes."""
    require(initial_offset >= 0, "initial field offset must be nonnegative")
    offsets = []
    field_names = set()
    offset = initial_offset
    for field in fields:
        valid = is_name(field.name) and field.name not in field_names
        require(valid and field.element_count > 0, f"{field.name}: invalid field")
        field_names.add(field.name)
        require(
            not offset % field.encoding.alignment,
            f"{field.name}: offset {offset} violates {field.encoding.alignment}-byte alignment; add explicit padding",
        )
        offsets.append(offset)
        offset += field.byte_length
    return tuple(offsets)
