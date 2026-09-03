# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM module numeric and flag domains."""

from iree.vm.bytecode.spec.schema import (
    U16,
    NumericKind,
    NumericTable,
    NumericValue,
    UnknownNumericPolicy,
)
from iree.vm.bytecode.spec.version import CORE_0


def _value(name: str, value: int, summary: str) -> NumericValue:
    return NumericValue(name, value, CORE_0, summary)


def _table(
    name: str,
    kind: NumericKind,
    values: tuple[tuple[str, int, str], ...],
    summary: str,
    unknown_policy: UnknownNumericPolicy = UnknownNumericPolicy.REJECT,
) -> NumericTable:
    return NumericTable(
        name,
        U16,
        kind,
        tuple(_value(*value) for value in values),
        CORE_0,
        summary,
        unknown_policy,
    )


SIGNATURE_KIND = _table(
    "signature_kind",
    NumericKind.ENUM,
    (
        ("invalid", 0x0000, "Invalid in a signature descriptor."),
        ("i8", 0x0001, "Low eight integer bits."),
        ("i16", 0x0002, "Low 16 integer bits."),
        ("i32", 0x0003, "Low 32 integer bits."),
        ("i64", 0x0004, "Complete 64 integer bits."),
        ("f8e4m3fn", 0x0005, "Low eight finite E4M3 floating-point bits."),
        ("f8e5m2", 0x0006, "Low eight E5M2 floating-point bits."),
        ("f16", 0x0007, "Low 16 IEEE binary16 bits."),
        ("bf16", 0x0008, "Low 16 bfloat16 bits."),
        ("f32", 0x0009, "Low 32 IEEE binary32 bits."),
        ("f64", 0x000A, "Complete 64 IEEE binary64 bits."),
        ("ref", 0x0100, "Exact ref type in the descriptor's ordinal field."),
        (
            "function",
            0x0200,
            "Exact callable type in the descriptor's ordinal field.",
        ),
    ),
    "Kinds encoded by source-ordered logical signature descriptors.",
)

SECTION_FLAG = _table(
    "section_flag",
    NumericKind.FLAGS,
    (("skippable", 1 << 0, "Unknown readers may skip the section."),),
    "Flags interpreted by a section's architectural authority.",
)

CALLABLE_TYPE_FLAG = _table(
    "callable_type_flag",
    NumericKind.FLAGS,
    (("may_yield", 1 << 0, "The callable contract permits suspension."),),
    "Callable behavior permissions used for structural compatibility.",
)

IMPORT_FLAG = _table(
    "import_flag",
    NumericKind.FLAGS,
    (("optional", 1 << 0, "An absent target module or export is permitted."),),
    "Import resolution behavior flags.",
)

FUNCTION_FLAG = _table(
    "function_flag",
    NumericKind.FLAGS,
    (
        ("may_yield", 1 << 0, "The function may yield."),
        ("has_call", 1 << 1, "The function contains a direct or indirect call."),
    ),
    "Statically derived bytecode function behavior flags.",
)

GLOBAL_REF_FLAG = _table(
    "global_ref_flag",
    NumericKind.FLAGS,
    (("nullable", 1 << 0, "The ref global may remain canonical null."),),
    "Reference-global initialization flags.",
)

GLOBAL_FUNCTION_FLAG = _table(
    "global_function_flag",
    NumericKind.FLAGS,
    (("nullable", 1 << 0, "The function global may remain canonical null."),),
    "Function-global initialization flags.",
)

PRESENTATION_DECLARATION_KIND = _table(
    "presentation_declaration_kind",
    NumericKind.ENUM,
    (
        ("invalid", 0, "Invalid in a presentation row."),
        ("import", 1, "An import declaration."),
        ("export", 2, "An export declaration."),
    ),
    "Public declaration domains carrying presentation data.",
)

METADATA_VALUE_TYPE = _table(
    "metadata_value_type",
    NumericKind.ENUM,
    (
        ("invalid", 0, "Invalid in a metadata entry."),
        ("bool", 1, "One canonical Boolean byte."),
        ("i64", 2, "Little-endian signed 64-bit bits."),
        ("u64", 3, "Little-endian unsigned 64-bit bits."),
        (
            "f64",
            4,
            "Little-endian IEEE binary64 bits with every bit pattern preserved.",
        ),
        ("utf8", 5, "Length-delimited UTF-8 bytes."),
        ("bytes", 6, "Opaque bytes."),
    ),
    "Typed metadata payload encodings.",
    UnknownNumericPolicy.PRESERVE_NONZERO,
)

NUMERIC_TABLES = (
    SIGNATURE_KIND,
    SECTION_FLAG,
    CALLABLE_TYPE_FLAG,
    IMPORT_FLAG,
    FUNCTION_FLAG,
    GLOBAL_REF_FLAG,
    GLOBAL_FUNCTION_FLAG,
    PRESENTATION_DECLARATION_KIND,
    METADATA_VALUE_TYPE,
)
