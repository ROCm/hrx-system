# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Closed declarations for VM module wire records."""

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.schema import Field, place_fields
from iree.vm.bytecode.spec.version import Version


class FieldRule(enum.Enum):
    ANY_BITS = "any_bits"
    ZERO = "zero"
    CORE_MAJOR = "core_major"
    CORE_REQUIRED_MINOR = "core_required_minor"


class ExactBytesRule(NamedTuple):
    expected: bytes


class AllowedRangeRule(NamedTuple):
    minimum: int
    maximum: int


WireFieldRule = FieldRule | ExactBytesRule | AllowedRangeRule


class WireField(NamedTuple):
    field: Field
    rule: WireFieldRule


class RecordRule(enum.Enum):
    SIGNATURE_DESCRIPTOR = "signature_descriptor"


class WireRecord(NamedTuple):
    """One naturally aligned fixed module wire record."""

    name: str
    c_type: str
    since: Version
    summary: str
    contract: str
    fields: tuple[WireField, ...]
    rules: tuple[RecordRule, ...] = ()

    @property
    def field_offsets(self) -> tuple[int, ...]:
        return place_fields(wire_field.field for wire_field in self.fields)

    @property
    def byte_length(self) -> int:
        return sum(field.field.byte_length for field in self.fields)

    @property
    def alignment(self) -> int:
        return max((field.field.encoding.alignment for field in self.fields), default=1)
