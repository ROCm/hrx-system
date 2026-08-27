# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C naming helpers for enum attributes in generated op metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from loom.dsl import AttrDef, EncodingFamilyDef, EnumDef, Op
from loom.gen.ops.c_names import c_encoding_enum_prefix, c_prefix
from loom.gen.support.c import CIdentifierCase, c_identifier

SharedEnumMap = dict[int, tuple[str, str, EnumDef]]
EncodingEnumMap = dict[int, tuple[EnumDef, bool]]
EncodingEnumTypeMap = dict[int, str]


def collect_encoding_enums(
    encoding_families: Sequence[EncodingFamilyDef],
) -> EncodingEnumMap:
    """Returns encoding-owned enums and whether each is open."""
    enums: EncodingEnumMap = {}
    auxiliary_key_enum: EnumDef | None = None
    for family in encoding_families:
        for parameter in family.parameters:
            enum_def = parameter.enum_def
            if parameter.attr_type not in ("enum", "enum_array", "signed_enum_set") or enum_def is None:
                continue
            previous = enums.get(id(enum_def))
            if previous is not None and previous[1] != parameter.open_enum:
                raise ValueError(f"encoding enum '{enum_def.name}' cannot be both open and closed")
            enums.setdefault(id(enum_def), (enum_def, parameter.open_enum))

        enum_def = family.auxiliary_key_enum
        if enum_def is None:
            continue
        if auxiliary_key_enum is not None and enum_def is not auxiliary_key_enum:
            raise ValueError("encoding families in one dialect must share one auxiliary key enum")
        auxiliary_key_enum = enum_def
        previous = enums.get(id(enum_def))
        if previous is not None and previous[1]:
            raise ValueError(f"auxiliary key enum '{enum_def.name}' must be closed")
        enums.setdefault(id(enum_def), (enum_def, False))
    return enums


def collect_encoding_auxiliary_key_enum(
    encoding_families: Sequence[EncodingFamilyDef],
) -> EnumDef | None:
    """Returns the one shared auxiliary-key enum, if present."""
    collect_encoding_enums(encoding_families)
    for family in encoding_families:
        if family.auxiliary_key_enum is not None:
            return family.auxiliary_key_enum
    return None


def collect_encoding_enum_types(
    dialect_name: str,
    encoding_families: Sequence[EncodingFamilyDef],
) -> EncodingEnumTypeMap:
    """Returns the canonical C type for encoding-owned enums."""
    return {enum_id: enum_def.c_type or f"{c_encoding_enum_prefix(dialect_name, enum_def)}_t" for enum_id, (enum_def, _) in collect_encoding_enums(encoding_families).items()}


def collect_shared_enums(
    dialect_name: str,
    ops: Sequence[Op],
) -> SharedEnumMap:
    """Identifies EnumDef objects shared across multiple ops in a dialect.

    Returns a mapping from id(enum_def) to (c_prefix, const_prefix, enum_def)
    for each EnumDef used by more than one op. The naming uses the dialect
    prefix + attr name (e.g., loom_func_cc for the func dialect's cc attr).

    EnumDefs used by only one op are not included; they keep per-op naming.
    """
    usage: dict[int, list[tuple[Op, AttrDef]]] = defaultdict(list)
    for op in ops:
        for attr_def in op.attrs:
            if attr_def.attr_type in ("enum", "enum_array", "signed_enum_set") and attr_def.enum_def is not None:
                if attr_def.enum_def.c_type is not None:
                    continue
                usage[id(attr_def.enum_def)].append((op, attr_def))

    shared: SharedEnumMap = {}
    for enum_id, pairs in usage.items():
        if len(pairs) <= 1:
            continue
        attr_names = {attr_def.name for _, attr_def in pairs}
        if len(attr_names) != 1:
            continue
        attr_name = next(iter(attr_names))
        c_name_prefix = f"loom_{dialect_name}_{attr_name}"
        const_prefix = c_name_prefix.upper()
        enum_def = pairs[0][1].enum_def
        assert enum_def is not None
        shared[enum_id] = (c_name_prefix, const_prefix, enum_def)

    return shared


def enum_c_prefix_from_type(c_type: str) -> str:
    """Returns the C prefix for an enum typedef name."""
    if c_type.endswith("_t"):
        return c_type[:-2]
    return c_type


def enum_c_prefix(
    op: Op,
    attr_def: AttrDef,
    shared_enums: SharedEnumMap,
) -> tuple[str, str]:
    """Returns (c_prefix, const_prefix) for an enum attr."""
    if attr_def.enum_def is not None and attr_def.enum_def.c_type is not None:
        assert attr_def.enum_def.c_const_prefix is not None
        return (
            enum_c_prefix_from_type(attr_def.enum_def.c_type),
            attr_def.enum_def.c_const_prefix,
        )
    if attr_def.enum_def is not None and id(attr_def.enum_def) in shared_enums:
        c_name_prefix, const_prefix, _ = shared_enums[id(attr_def.enum_def)]
        return c_name_prefix, const_prefix
    c_name_prefix = c_prefix(op) + "_" + attr_def.name
    return c_name_prefix, c_name_prefix.upper()


def enum_c_type(
    op: Op,
    attr_def: AttrDef,
    shared_enums: SharedEnumMap,
) -> str:
    """Returns the C type exposed for an enum attribute."""
    if attr_def.enum_def is not None and attr_def.enum_def.c_type is not None:
        return attr_def.enum_def.c_type
    c_name_prefix, _ = enum_c_prefix(op, attr_def, shared_enums)
    return f"{c_name_prefix}_t"


def enum_names_array_name(
    op: Op,
    attr_def: AttrDef,
    shared_enums: SharedEnumMap,
) -> str:
    """Returns the enum keyword table symbol for an enum attribute."""
    if attr_def.enum_def is not None and attr_def.enum_def.c_type is not None:
        c_name_prefix, _ = enum_c_prefix(op, attr_def, shared_enums)
        return f"{c_name_prefix}_names"
    shared = shared_enums.get(id(attr_def.enum_def)) if attr_def.enum_def is not None else None
    if shared:
        return f"{shared[0]}_names"
    return f"{c_prefix(op)}_{attr_def.name}_names"


def enum_case_c_ident(keyword: str) -> str:
    """Converts an enum assembly keyword to a C enum/macro suffix."""
    return c_identifier(keyword, case=CIdentifierCase.UPPER)
