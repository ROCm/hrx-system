# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C naming helpers for generated op tables."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from loom.dsl import EncodingFamilyDef, EnumDef, Op, ParameterizedAttrDef
from loom.gen.support.generated_file import line_comment_header

COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

GENERATED_HEADER = COPYRIGHT + "\n" + "\n".join(line_comment_header("//", generator="loom.gen.ops.c_tables")) + "\n// clang-format off"


def c_prefix(op: Op) -> str:
    """Returns the C function/variable prefix for an op.

    test.addi -> loom_test_addi
    """
    return "loom_" + op.name.replace(".", "_")


def c_enum_name(op: Op) -> str:
    """Returns the C enum constant name for an op kind.

    test.addi -> LOOM_OP_TEST_ADDI
    """
    return "LOOM_OP_" + op.name.replace(".", "_").upper()


def c_parameterized_attr_prefix(attr_def: ParameterizedAttrDef) -> str:
    """Returns the C function/variable prefix for an attribute family."""

    return "loom_" + attr_def.name.replace(".", "_")


def c_parameterized_attr_api_prefix(attr_def: ParameterizedAttrDef) -> str:
    """Returns the C API prefix for a parameterized attribute family."""

    return c_parameterized_attr_prefix(attr_def) + "_attr"


def c_parameterized_attr_enum_name(attr_def: ParameterizedAttrDef) -> str:
    """Returns the C enum constant for a parameterized attribute family."""

    return "LOOM_PARAMETERIZED_ATTR_" + attr_def.name.replace(".", "_").upper()


def c_encoding_family_prefix(family: EncodingFamilyDef) -> str:
    """Returns the C function/variable prefix for an encoding family."""

    local_name = family.name.removeprefix(f"{family.group.name}.")
    return "loom_encoding_" + local_name.replace(".", "_")


def validate_encoding_family_c_names(
    families: Sequence[EncodingFamilyDef],
) -> None:
    """Rejects duplicate or colliding generated encoding-family names."""

    families_by_prefix: dict[str, EncodingFamilyDef] = {}
    public_names: dict[str, str] = {}
    for family in families:
        prefix = c_encoding_family_prefix(family)
        previous_family = families_by_prefix.get(prefix)
        if previous_family is not None:
            if previous_family.name == family.name:
                raise ValueError(f"duplicate encoding family '{family.name}'")
            raise ValueError(f"encoding families '{previous_family.name}' and '{family.name}' both generate C prefix '{prefix}'")
        families_by_prefix[prefix] = family
        for public_name in (family.name, *(alias.name for alias in family.aliases)):
            previous_owner = public_names.get(public_name)
            if previous_owner is not None:
                raise ValueError(f"encoding name '{public_name}' is declared by both '{previous_owner}' and '{family.name}'")
            public_names[public_name] = family.name


def c_encoding_family_descriptor_name(family: EncodingFamilyDef) -> str:
    """Returns the exported C descriptor symbol for an encoding family."""

    return c_encoding_family_prefix(family) + "_family_descriptor"


def c_encoding_enum_prefix(dialect_name: str, enum_def: EnumDef) -> str:
    """Returns the shared C prefix for an encoding-family parameter enum."""

    snake_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", enum_def.name)
    snake_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_name).lower()
    return f"loom_{dialect_name}_{snake_name}"


def c_dialect_enum(dialect_name: str) -> str:
    """Returns the C dialect ID enum name.

    test -> LOOM_DIALECT_TEST
    """
    return "LOOM_DIALECT_" + dialect_name.upper()


def c_dialect_path(dialect: Any) -> str:
    """Returns the generated C source path under loom/src/loom."""
    return dialect.c_path or f"ops/{dialect.name}"


def c_dialect_include_path(dialect: Any) -> str:
    """Returns the generated C include path rooted at loom/src."""
    return f"loom/{c_dialect_path(dialect)}"


def guard_name(dialect_name: str) -> str:
    """Returns the generated ops.h include guard name."""
    return f"LOOM_OPS_{dialect_name.upper()}_OPS_H_"
