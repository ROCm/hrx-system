# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C spelling helpers for target-low descriptor tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from loom.gen.support.c import CIdentifierCase
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.string_pool import CStringPool
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    CEnum,
    DescriptorSet,
    descriptor_set_relative_name,
)


def c_identifier(value: str) -> str:
    return _c_identifier(value, case=CIdentifierCase.LOWER, empty="empty")


def flag_expr(flags: Iterable[CEnum]) -> str:
    names = [flag.c_name for flag in flags]
    return " | ".join(names) if names else "0"


def optional_string_expr(string_pool: CStringPool, label: str | None) -> str:
    if label is None:
        return "LOOM_LOW_STRING_OFFSET_NONE"
    return string_pool.ref(label)


def i64_literal(value: int) -> str:
    if value == -(1 << 63):
        return "INT64_MIN"
    if value < 0:
        return f"(-INT64_C({abs(value)}))"
    return f"INT64_C({value})"


def u64_literal(value: int) -> str:
    return f"UINT64_C({value})"


def hex_u64_literal(value: int) -> str:
    return f"UINT64_C(0x{value:x})"


def hex_u32_literal(value: int) -> str:
    return f"UINT32_C(0x{value:x})"


def u16_literal(value: int) -> str:
    return f"UINT16_C({value})"


def descriptor_ref_constant_name(spec: DescriptorSet, descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(spec, descriptor_key)
    return f"{spec.c_enum_prefix}_DESCRIPTOR_REF_{c_identifier(descriptor_name).upper()}"


def descriptor_ref_define(spec: DescriptorSet, descriptor_key: str, descriptor_ordinal: int) -> str:
    return f"#define {descriptor_ref_constant_name(spec, descriptor_key)} {descriptor_ordinal}u"


def reg_class_id_constant_name(spec: DescriptorSet, reg_class_name: str) -> str:
    reg_class_name = descriptor_set_relative_name(spec, reg_class_name)
    return f"{spec.c_enum_prefix}_REG_CLASS_ID_{c_identifier(reg_class_name).upper()}"


def register_part_id_constant_name(spec: DescriptorSet, part_name: str) -> str:
    part_name = descriptor_set_relative_name(spec, part_name)
    return f"{spec.c_enum_prefix}_REGISTER_PART_ID_{c_identifier(part_name).upper()}"


def emit_id_enum(typedef_name: str, entries: Sequence[tuple[str, int]]) -> list[str]:
    if not entries:
        return []
    lines = [f"typedef enum {typedef_name}_e {{"]
    lines.extend(f"  {name} = {value}u," for name, value in entries)
    lines.append(f"}} {typedef_name}_t;")
    return lines


def encoding_id_expr(value: int) -> str:
    return "LOOM_LOW_ID_NONE" if value == LOW_DESCRIPTOR_ENCODING_ID_NONE else str(value)


def canonical_asm_form_ordinal_expr(value: int | None) -> str:
    if value is None:
        return "LOOM_LOW_ASM_FORM_ORDINAL_NONE"
    return str(value)
