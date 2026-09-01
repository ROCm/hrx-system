# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 module-format numeric and flag domains."""

from __future__ import annotations

from model.schema import (
    U16,
    NumericTable,
    NumericTableKind,
    NumericValue,
    ScalarEncoding,
    UnknownNumericValuePolicy,
)
from model.specification import CORE_0


def _table(
    key: str,
    encoding: ScalarEncoding,
    table_kind: NumericTableKind,
    values: tuple[tuple[str, int, str], ...],
    *,
    unknown_value_policy: UnknownNumericValuePolicy = (
        UnknownNumericValuePolicy.REJECT
    ),
) -> tuple[NumericTable, tuple[NumericValue, ...]]:
    table_id = f"core.module.numeric.{key}"
    table = NumericTable(
        entity_id=table_id,
        since=CORE_0,
        summary=f"Module-format {key.replace('_', ' ')} values.",
        encoding_id=encoding.entity_id,
        table_kind=table_kind,
        unknown_value_policy=unknown_value_policy,
    )
    return table, tuple(
        NumericValue(
            entity_id=f"{table_id}.{name}",
            since=CORE_0,
            summary=description,
            minimum_consumer_version=(
                table.since
                if unknown_value_policy == UnknownNumericValuePolicy.PRESERVE_NONZERO
                else None
            ),
            table_id=table_id,
            name=name,
            value=value,
        )
        for name, value, description in values
    )


_TABLE_DEFINITIONS = (
    _table(
        "signature_kind",
        U16,
        NumericTableKind.ENUM,
        (
            ("invalid", 0x0000, "Invalid in a descriptor."),
            ("i8", 0x0001, "Low eight integer bits."),
            ("i16", 0x0002, "Low 16 integer bits."),
            ("i32", 0x0003, "Low 32 integer bits."),
            ("i64", 0x0004, "Complete 64 integer bits."),
            ("f8e4m3fn", 0x0005, "Low eight floating-point bits."),
            ("f8e5m2", 0x0006, "Low eight floating-point bits."),
            ("f16", 0x0007, "Low 16 IEEE binary16 bits."),
            ("bf16", 0x0008, "Low 16 bfloat16 bits."),
            ("f32", 0x0009, "Low 32 IEEE binary32 bits."),
            ("f64", 0x000A, "Complete 64 IEEE binary64 bits."),
            ("ref", 0x0100, "Exact ref type in the second field."),
            (
                "function",
                0x0200,
                "Exact callable type in the second field.",
            ),
        ),
    ),
    _table(
        "section_flag",
        U16,
        NumericTableKind.FLAGS,
        (("skippable", 1 << 0, "Unknown readers may skip the section."),),
    ),
    _table(
        "callable_type_flag",
        U16,
        NumericTableKind.FLAGS,
        (
            (
                "may_yield",
                1 << 0,
                "The callable contract permits suspension.",
            ),
        ),
    ),
    _table(
        "import_flag",
        U16,
        NumericTableKind.FLAGS,
        (
            (
                "optional",
                1 << 0,
                "An absent target module or export is permitted.",
            ),
        ),
    ),
    _table(
        "function_flag",
        U16,
        NumericTableKind.FLAGS,
        (
            ("may_yield", 1 << 0, "The function may yield."),
            (
                "has_call",
                1 << 1,
                "The function contains a direct or indirect call.",
            ),
        ),
    ),
    _table(
        "global_ref_flag",
        U16,
        NumericTableKind.FLAGS,
        (
            (
                "nullable",
                1 << 0,
                "The ref global may remain canonical null.",
            ),
        ),
    ),
    _table(
        "global_function_flag",
        U16,
        NumericTableKind.FLAGS,
        (
            (
                "nullable",
                1 << 0,
                "The function global may remain canonical null.",
            ),
        ),
    ),
    _table(
        "presentation_declaration_kind",
        U16,
        NumericTableKind.ENUM,
        (
            ("invalid", 0, "Invalid in a presentation row."),
            ("import", 1, "An import declaration."),
            ("export", 2, "An export declaration."),
        ),
    ),
    _table(
        "metadata_value_type",
        U16,
        NumericTableKind.ENUM,
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
        unknown_value_policy=UnknownNumericValuePolicy.PRESERVE_NONZERO,
    ),
)

NUMERIC_TABLES = tuple(definition[0] for definition in _TABLE_DEFINITIONS)
NUMERIC_VALUES = tuple(
    value for definition in _TABLE_DEFINITIONS for value in definition[1]
)
NUMERIC_TABLES_BY_KEY = {
    table.entity_id.removeprefix("core.module.numeric."): table
    for table in NUMERIC_TABLES
}
ENTITIES = (*NUMERIC_TABLES, *NUMERIC_VALUES)
